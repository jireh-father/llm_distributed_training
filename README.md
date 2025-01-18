# LLM Distributed Training

This repository contains code for distributed training of large language models using various parallelization techniques.

## Features

- Support for multiple distributed training methods:
  - DeepSpeed ZeRO-3
  - FSDP (Fully Sharded Data Parallel)
  - Tensor Parallelism
  - Pipeline Parallelism

- PEFT (Parameter-Efficient Fine-Tuning) methods:
  - LoRA (Low-Rank Adaptation)
  - Prefix Tuning
  - Prompt Tuning
  - P-Tuning v2

- Supported Models:
  - Gemma (2B/7B/27B)
  - Qwen2 (72B)

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### Single GPU Training with LoRA (Specific GPU)

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model_name_or_path="google/gemma-2-2b-it" --distributed_type="no" --peft_type="lora" --batch_size=1 --gradient_accumulation_steps=16 --learning_rate=1e-5 --num_epochs=3 --max_length=8192 --lora_r=16 --lora_alpha=32 --lora_dropout=0.1 --lora_target_modules="q_proj,v_proj" --output_dir="output" --cache_dir="./cache" --hf_token="your_huggingface_token"
```

### Single GPU Training with LoRA (Default GPU)

```bash
python train.py --model_name_or_path="google/gemma-2b-it" --distributed_type="no" --peft_type="lora" --batch_size=1 --gradient_accumulation_steps=16 --learning_rate=1e-5 --num_epochs=3 --max_length=128 --lora_r=16 --lora_alpha=32 --lora_dropout=0.1 --lora_target_modules="q_proj,v_proj" --output_dir="output" --cache_dir="./cache" --hf_token="your_huggingface_token"
```

### Multi-GPU Training with DeepSpeed ZeRO

```bash
torchrun --nproc_per_node=4 train.py --model_name_or_path="google/gemma-27b-it" --distributed_type="deepspeed" --peft_type="lora" --batch_size=1 --gradient_accumulation_steps=16 --learning_rate=1e-5 --num_epochs=3 --max_length=128 --lora_r=16 --lora_alpha=32 --lora_dropout=0.1 --lora_target_modules="q_proj,v_proj" --output_dir="output" --cache_dir="./cache" --hf_token="your_huggingface_token"
```

### Pipeline Parallel Training with 2 GPUs

```bash
torchrun --nproc_per_node=2 train.py --model_name_or_path="google/gemma-2-27b-it" --distributed_type="deepspeed_pp" --pipeline_parallel_size=2 --pipeline_chunk_size=1 --peft_type="lora" --batch_size=1 --gradient_accumulation_steps=16 --learning_rate=1e-5 --num_epochs=3 --max_length=128 --lora_r=16 --lora_alpha=32 --lora_dropout=0.1 --lora_target_modules="q_proj,v_proj" --output_dir="output" --cache_dir="./cache" --hf_token="your_huggingface_token"
```

## Configuration

- `distributed_type`: Choose distributed training method ("no", "deepspeed", "fsdp")
- `peft_type`: Choose PEFT method ("lora", "prefix", "prompt", "p-tuning")
- `batch_size`: Micro batch size per GPU
- `gradient_accumulation_steps`: Number of steps for gradient accumulation
- `lora_r`: LoRA rank
- `lora_alpha`: LoRA scaling factor
- `lora_target_modules`: Target modules for LoRA adaptation

## Memory Optimization

- Gradient Checkpointing
- Mixed Precision Training (FP16)
- CPU Offloading (optional)
- 8-bit Optimizer (bitsandbytes)

## Notes

- For Gemma models, you need to accept the model license and provide a Hugging Face token
- Adjust batch size and gradient accumulation steps based on your GPU memory
- Choose appropriate parallelization strategy based on model size and available GPUs 