# Step-Audio-2 端到端 SFT 微调

该目录提供 `finetune/train_sft.py`，满足以下能力：

- 训练目标：`llm` / `audio_detokenizer`
- 微调类型：`full` / `lora`
- 分布式：单机多卡、多机多卡、DeepSpeed(ZeRO2/ZeRO3)
- 参数可配：`lr`、`batch`、`gradient_accumulation_steps`、精度（`bf16/fp16/fp32`）等
- 支持你当前数据范式：`输入audio + (可选history text) + 输出audio + think`
- 支持“LLM + tokenizer 同时微调”（通过模块名匹配解冻）

---

## 1. 安装

```bash
pip install -U torch transformers datasets peft deepspeed accelerate
```

如果训练 `llm` 且含音频输入，还需要本仓库推理依赖（`torchaudio/librosa/s3tokenizer` 等）。

---

## 2. 数据格式

## 2.1 推荐：audio_text_think（你的场景）

训练命令中加：`--data_format audio_text_think`

每条样本（json/jsonl）示例：

```json
{
  "system": "You are a helpful assistant.",
  "input_audio": "/path/current_input.wav",
  "input_text": "可选文本补充（例如指令）",
  "history": [
    {
      "input_audio_text": "上一轮用户音频转写",
      "output_think": "上一轮assistant思考",
      "output_audio_text": "上一轮assistant语音转写"
    }
  ],
  "output_think": "本轮思考过程文本",
  "output_audio_text": "本轮语音内容转写（可选）",
  "output_audio_tokens": [1493, 4299, 4218]
}
```

字段说明：
- `input_audio`：必填，当前轮输入音频路径。
- `history`：可选，历史信息（你提到的历史输入音频转录+历史输出think+历史输出音频转录）。
- `output_think`：必填，监督模型输出思考文本。
- `output_audio_tokens`：可选，如你有离散音频 token，可直接监督到 assistant 输出。
- `output_audio_text`：可选，可作为 transcript 文本监督。

> 注意：当前 Step-Audio-2 接口下，含音频训练建议 `--per_device_train_batch_size 1`（脚本中对多样本音频 batch 做了保护）。

### 2.2 通用 messages 格式

训练命令中加：`--data_format messages`（默认）。

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "human", "content": [
      {"type": "text", "text": "请转写这段语音"},
      {"type": "audio", "audio": "assets/mmau_test.wav"}
    ]},
    {"role": "assistant", "content": "转写结果..."}
  ]
}
```

---

## 3. LLM + tokenizer 同时微调

> 这里的 tokenizer 指模型内可训练 tokenizer / speech-tokenizer 相关子模块（按参数名匹配）。

关键参数：
- `--joint_train_tokenizer true`
- `--tokenizer_trainable_patterns "audio_tokenizer,speech_tokenizer,tokenizer"`

脚本会对这些名称命中的参数解冻训练。

---

## 4. 单机多卡（LoRA）

```bash
torchrun --nproc_per_node 8 finetune/train_sft.py \
  --model_name_or_path stepfun-ai/Step-Audio-2-mini \
  --train_target llm \
  --data_format audio_text_think \
  --finetune_type lora \
  --joint_train_tokenizer true \
  --tokenizer_trainable_patterns audio_tokenizer,speech_tokenizer,tokenizer \
  --train_file /path/to/train.jsonl \
  --validation_file /path/to/valid.jsonl \
  --max_length 4096 \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --weight_decay 0.1 \
  --bf16 true \
  --tf32 true \
  --logging_steps 10 \
  --save_steps 500 \
  --eval_steps 500 \
  --evaluation_strategy steps \
  --save_strategy steps \
  --save_total_limit 3 \
  --output_dir /path/to/output
```

如需 full 参数，把 `--finetune_type` 改为 `full`。

---

## 5. DeepSpeed

```bash
torchrun --nproc_per_node 8 finetune/train_sft.py \
  --model_name_or_path stepfun-ai/Step-Audio-2-mini \
  --train_target llm \
  --data_format audio_text_think \
  --finetune_type lora \
  --joint_train_tokenizer true \
  --train_file /path/to/train.jsonl \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --bf16 true \
  --deepspeed finetune/configs/deepspeed_zero2.json \
  --output_dir /path/to/output
```

可替换为 `deepspeed_zero3.json`。

---

## 6. 多机多卡

```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=29500 \
  finetune/train_sft.py \
  --model_name_or_path stepfun-ai/Step-Audio-2-mini \
  --train_target llm \
  --data_format audio_text_think \
  --finetune_type lora \
  --joint_train_tokenizer true \
  --train_file /path/to/train.jsonl \
  --learning_rate 1e-4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --bf16 true \
  --deepspeed finetune/configs/deepspeed_zero2.json \
  --output_dir /path/to/output
```

---

## 7. AudioDetokenizer 训练

```bash
torchrun --nproc_per_node 8 finetune/train_sft.py \
  --train_target audio_detokenizer \
  --audio_detokenizer_model_name_or_path /path/to/audio-detok-model \
  --finetune_type full \
  --train_file /path/to/audio_detok_train.jsonl \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --bf16 true \
  --output_dir /path/to/audio_detok_output
```
