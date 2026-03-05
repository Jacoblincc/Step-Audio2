#!/usr/bin/env python3
"""Step-Audio-2 SFT training entrypoint.

Supports:
- SFT target: llm / audio_detokenizer
- fine-tune type: full / lora
- optional joint training for tokenizer-related submodules inside model
- single-node multi-GPU / multi-node multi-GPU / deepspeed (through TrainingArguments)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from utils import compute_token_num, load_audio, log_mel_spectrogram, padding_mels

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="stepfun-ai/Step-Audio-2-mini")
    train_target: Literal["llm", "audio_detokenizer"] = field(default="llm")
    audio_detokenizer_model_name_or_path: Optional[str] = field(default=None)
    trust_remote_code: bool = field(default=True)
    attn_implementation: Optional[str] = field(default=None)
    torch_dtype: Literal["auto", "bf16", "fp16", "fp32"] = field(default="auto")


@dataclass
class DataArguments:
    train_file: str = field(metadata={"help": "json/jsonl training data"})
    validation_file: Optional[str] = field(default=None)
    max_length: int = field(default=4096)
    preprocessing_num_workers: int = field(default=4)
    data_format: Literal["messages", "audio_text_think"] = field(
        default="messages",
        metadata={"help": "messages=原始对话格式; audio_text_think=输入audio+history+输出audio+think格式"},
    )


@dataclass
class FinetuneArguments:
    finetune_type: Literal["full", "lora"] = field(default="lora")
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # Joint train option for "LLM + tokenizer".
    joint_train_tokenizer: bool = field(
        default=False,
        metadata={"help": "同时训练模型中 tokenizer/audio tokenizer 相关子模块（按名称匹配）"},
    )
    tokenizer_trainable_patterns: str = field(
        default="audio_tokenizer,speech_tokenizer,tokenizer",
        metadata={"help": "逗号分隔，匹配需要解冻训练的 tokenizer 子模块名"},
    )


@dataclass
class ScriptArguments:
    seed: int = field(default=42)


def _to_torch_dtype(dtype: str):
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "auto": "auto",
    }
    return mapping[dtype]


def _chunk_audio_to_tokens(audio_path: str) -> tuple[int, List[torch.Tensor]]:
    mels: List[torch.Tensor] = []
    token_count = 0
    audio = load_audio(audio_path)
    for i in range(0, audio.shape[0], 16000 * 25):
        mel = log_mel_spectrogram(audio[i : i + 16000 * 25], n_mels=128, padding=479)
        mels.append(mel)
        token_count += compute_token_num(mel.shape[1])
    return token_count, mels


def _render_messages(messages: Sequence[Dict[str, Any]]) -> tuple[List[str | List[int]], List[torch.Tensor]]:
    results: List[str | List[int]] = []
    mels: List[torch.Tensor] = []

    for msg in messages:
        role = "human" if msg["role"] == "user" else msg["role"]
        content = msg.get("content")

        if isinstance(content, str):
            text = f"<|BOT|>{role}\n{content}"
            if msg.get("eot", True):
                text += "<|EOT|>"
            results.append(text)
        elif isinstance(content, list):
            results.append(f"<|BOT|>{role}\n")
            for item in content:
                item_type = item["type"]
                if item_type == "text":
                    results.append(item["text"])
                elif item_type == "token":
                    results.append(item["token"])
                elif item_type == "audio":
                    token_num, item_mels = _chunk_audio_to_tokens(item["audio"])
                    mels.extend(item_mels)
                    results.append("<audio_start>" + "<audio_patch>" * token_num + "<audio_end>")
                else:
                    raise ValueError(f"Unsupported content type: {item_type}")
            if msg.get("eot", True):
                results.append("<|EOT|>")
        elif content is None:
            results.append(f"<|BOT|>{role}\n")
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

    return results, mels


def _build_messages_from_audio_text_think(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    """将用户描述的数据格式转成 messages。

    输入样例（建议）：
    {
      "system": "...",                     # 可选
      "input_audio": "path.wav",           # 必填
      "input_text": "...",                 # 可选
      "history": [                          # 可选
        {
          "input_audio_text": "...",
          "output_think": "...",
          "output_audio_text": "..."
        }
      ],
      "output_think": "...",               # 必填
      "output_audio_text": "...",          # 可选
      "output_audio_tokens": [1,2,3]        # 可选，若提供会并入 assistant 监督
    }
    """
    messages: List[Dict[str, Any]] = []

    system_prompt = example.get("system")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for hist in example.get("history", []):
        hist_user = hist.get("input_audio_text", "")
        messages.append({"role": "human", "content": hist_user})

        think_text = hist.get("output_think", "")
        audio_text = hist.get("output_audio_text", "")
        assistant_text = think_text
        if audio_text:
            assistant_text = f"{assistant_text}\n<audio_transcript>{audio_text}</audio_transcript>"
        messages.append({"role": "assistant", "content": assistant_text})

    user_content: List[Dict[str, Any]] = [{"type": "audio", "audio": example["input_audio"]}]
    if example.get("input_text"):
        user_content.insert(0, {"type": "text", "text": example["input_text"]})
    messages.append({"role": "human", "content": user_content})

    assistant_items: List[Dict[str, Any]] = []
    if example.get("output_think"):
        assistant_items.append({"type": "text", "text": f"<think>{example['output_think']}</think>"})
    if example.get("output_audio_text"):
        assistant_items.append(
            {"type": "text", "text": f"\n<audio_transcript>{example['output_audio_text']}</audio_transcript>"}
        )
    if example.get("output_audio_tokens"):
        assistant_items.append({"type": "token", "token": example["output_audio_tokens"]})

    messages.append({"role": "assistant", "content": assistant_items})
    return messages


def build_llm_preprocess_fn(tokenizer: AutoTokenizer, max_length: int, data_format: str):
    def _preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        if data_format == "audio_text_think":
            messages = _build_messages_from_audio_text_think(example)
        else:
            messages = example["messages"]

        fragments, mels = _render_messages(messages)
        token_ids: List[int] = []
        for frag in fragments:
            if isinstance(frag, str):
                token_ids.extend(tokenizer(frag, add_special_tokens=False)["input_ids"])
            else:
                token_ids.extend(frag)

        token_ids = token_ids[:max_length]
        labels = token_ids.copy()
        if "loss_mask" in example:
            mask = example["loss_mask"][: len(labels)]
            labels = [token if m else -100 for token, m in zip(labels, mask)]

        return {
            "input_ids": token_ids,
            "labels": labels,
            "wavs": [mel.tolist() for mel in mels],
        }

    return _preprocess


def build_audio_detok_preprocess_fn(tokenizer: AutoTokenizer, max_length: int):
    def _preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        prompt_ids = tokenizer(example.get("prompt", ""), add_special_tokens=False)["input_ids"]
        audio_tokens = example["audio_tokens"]
        seq = (prompt_ids + audio_tokens)[:max_length]
        labels = [-100] * min(len(prompt_ids), len(seq)) + seq[len(prompt_ids) :]
        return {"input_ids": seq, "labels": labels[: len(seq)]}

    return _preprocess


class SFTDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []

        for feat in features:
            pad = max_len - len(feat["input_ids"])
            input_ids.append(feat["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append([1] * len(feat["input_ids"]) + [0] * pad)
            labels.append(feat["labels"] + [-100] * pad)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        has_audio = features[0].get("wavs") is not None
        if has_audio:
            # 对于 Step-Audio-2 当前接口，wavs 与文本 token 对齐逻辑是按样本顺序拼接。
            # 为避免跨样本音频错配，含音频训练建议使用 batch_size=1。
            if len(features) > 1 and any(f.get("wavs") for f in features):
                raise ValueError("Audio SFT currently requires per_device_train_batch_size=1 to keep wav/text alignment.")

            wavs_raw = features[0].get("wavs") or []
            if wavs_raw:
                mel_tensors = [torch.tensor(w, dtype=torch.float32) for w in wavs_raw]
                wavs, wav_lens = padding_mels(mel_tensors)
                batch["wavs"] = wavs
                batch["wav_lens"] = wav_lens

        return batch


def _enable_tokenizer_joint_train(model: torch.nn.Module, patterns: List[str]):
    hit = 0
    for name, param in model.named_parameters():
        if any(p in name for p in patterns):
            param.requires_grad = True
            hit += 1
    logger.info("Tokenizer joint training enabled, matched params: %d", hit)


def maybe_apply_lora(model, ft_args: FinetuneArguments):
    if ft_args.finetune_type == "full":
        return model

    target_modules = [x.strip() for x in ft_args.lora_target_modules.split(",") if x.strip()]
    lora_cfg = LoraConfig(
        r=ft_args.lora_r,
        lora_alpha=ft_args.lora_alpha,
        lora_dropout=ft_args.lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def load_model_and_tokenizer(model_args: ModelArguments):
    if model_args.train_target == "audio_detokenizer":
        model_path = model_args.audio_detokenizer_model_name_or_path
        if not model_path:
            raise ValueError("audio_detokenizer_model_name_or_path is required when train_target=audio_detokenizer")
    else:
        model_path = model_args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=model_args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=_to_torch_dtype(model_args.torch_dtype),
        attn_implementation=model_args.attn_implementation,
    )
    return model, tokenizer


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, FinetuneArguments, TrainingArguments, ScriptArguments))
    model_args, data_args, ft_args, training_args, script_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if training_args.local_rank in (-1, 0) else logging.WARN,
    )
    set_seed(script_args.seed)

    model, tokenizer = load_model_and_tokenizer(model_args)
    model = maybe_apply_lora(model, ft_args)

    if ft_args.joint_train_tokenizer:
        patterns = [x.strip() for x in ft_args.tokenizer_trainable_patterns.split(",") if x.strip()]
        _enable_tokenizer_joint_train(model, patterns)

    data_files = {"train": data_args.train_file}
    if data_args.validation_file:
        data_files["validation"] = data_args.validation_file
    raw_datasets: DatasetDict = load_dataset("json", data_files=data_files)

    if model_args.train_target == "llm":
        preprocess_fn = build_llm_preprocess_fn(tokenizer, data_args.max_length, data_args.data_format)
    else:
        preprocess_fn = build_audio_detok_preprocess_fn(tokenizer, data_args.max_length)

    processed = raw_datasets.map(
        preprocess_fn,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing datasets",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed["train"],
        eval_dataset=processed.get("validation"),
        tokenizer=tokenizer,
        data_collator=SFTDataCollator(pad_token_id=tokenizer.pad_token_id),
    )

    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    if training_args.do_eval and "validation" in processed:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
