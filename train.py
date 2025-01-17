import argparse
import os
import evaluate
import torch
import deepspeed
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from accelerate.state import AcceleratorState
from accelerate.utils.dataclasses import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    BackwardPrefetch,
    StateDictType,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from datasets import load_dataset
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptEncoderConfig,
    TaskType
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
    AutoConfig
)
from torch.utils.data import DataLoader
from typing import Dict, List
import bitsandbytes as bnb

def parse_args():
    parser = argparse.ArgumentParser(description="파인튜닝 스크립트")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="google/gemma-2-27b-it",
        choices=[
            "google/gemma-2-2b-it",
            "google/gemma-2-9b-it",
            "google/gemma-2-27b-it",
            "Qwen/Qwen2.5-72B-Instruct"
        ],
        help="사용할 모델 선택 (Gemma 9B/27B 또는 Qwen 72B)"
    )
    parser.add_argument(
        "--peft_type",
        type=str,
        default="lora",
        choices=["lora", "prefix", "prompt", "p-tuning"],
        help="사용할 PEFT 방식"
    )
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=42)
    
    # 모델 병렬화 관련 인자
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=2,
        help="Tensor Parallelism 크기"
    )
    parser.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=2,
        help="Pipeline Parallelism 크기"
    )
    parser.add_argument(
        "--pipe_chunk_size",
        type=int,
        default=2,
        help="Pipeline parallel chunk 크기"
    )
    
    # LoRA 관련 인자
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj")
    
    # Prefix Tuning 관련 인자
    parser.add_argument("--num_virtual_tokens", type=int, default=20)
    parser.add_argument("--encoder_hidden_size", type=int, default=512)
    
    # 캐시 관련 인자 추가
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="모델과 토크나이저의 캐시 저장 경로"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./dataset",
        help="데이터셋 저장 경로"
    )
    
    # 분산 학습 관련 인자 추가
    parser.add_argument(
        "--distributed_type",
        type=str,
        default="fsdp",
        choices=["no", "deepspeed", "fsdp"],
        help="분산 학습 방식 선택"
    )
    parser.add_argument(
        "--fsdp_offload",
        action="store_true",
        help="FSDP CPU 오프로딩 사용"
    )
    
    # Hugging Face 토큰 설정
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face 토큰 (필수: Gemma 모델 접근용)"
    )
    
    return parser.parse_args()

def get_peft_config(peft_type: str, args: argparse.Namespace):
    if peft_type == "lora":
        return LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules.split(","),
            bias="none",
        )
    elif peft_type == "prefix":
        return PrefixTuningConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=args.num_virtual_tokens,
        )
    elif peft_type == "prompt":
        return PromptTuningConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=args.num_virtual_tokens,
        )
    elif peft_type == "p-tuning":
        return PromptEncoderConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=args.num_virtual_tokens,
            encoder_hidden_size=args.encoder_hidden_size,
        )
    raise ValueError(f"Unknown PEFT type: {peft_type}")

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 분산 학습 환경 설정
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://"
        )
    
    # Gemma 모델일 경우 토큰 검증
    if "gemma" in args.model_name_or_path and args.hf_token is None:
        raise ValueError(
            "Gemma 모델에 접근하기 위해서는 Hugging Face 토큰이 필요합니다. "
            f"1. https://huggingface.co/{args.model_name_or_path} 에서 모델 사용 약관에 동의하세요. "
            "2. https://huggingface.co/settings/tokens 에서 토큰을 생성하세요. "
            "3. --hf_token 인자로 토큰을 전달하세요."
        )
    
    # 캐시 디렉토리 생성 (메인 프로세스에서만)
    if local_rank in [-1, 0]:
        os.makedirs(args.cache_dir, exist_ok=True)
        os.makedirs(args.dataset_dir, exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)
    if world_size > 1:
        torch.distributed.barrier()  # 다른 프로세스들이 디렉토리 생성을 기다림
    
    # 환경변수 설정
    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_DATASETS_CACHE"] = os.path.join(args.cache_dir, "datasets")
    if args.hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = args.hf_token
    
    # 분산 학습 플러그인 설정
    if args.distributed_type == "deepspeed":
        plugin = DeepSpeedPlugin(
            hf_ds_config={
                "zero_optimization": {
                    "stage": 3,
                    "offload_optimizer": {
                        "device": "none",
                        "pin_memory": False
                    },
                    "offload_param": {
                        "device": "none",
                        "pin_memory": False
                    },
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "stage3_gather_16bit_weights_on_model_save": True,
                    "reduce_bucket_size": int(5e8),
                    "stage3_param_persistence_threshold": int(1e6)
                },
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "gradient_clipping": 1.0,
                "train_batch_size": args.batch_size * args.gradient_accumulation_steps * world_size,
                "train_micro_batch_size_per_gpu": args.batch_size,
                "fp16": {
                    "enabled": True,
                    "loss_scale": 0,
                    "loss_scale_window": 1000,
                    "initial_scale_power": 16,
                    "hysteresis": 2,
                    "min_loss_scale": 1
                },
                "tensor_parallel": {
                    "enabled": True,
                    "size": 2  # 2-way tensor parallelism
                },
                "pipeline_parallel": {
                    "enabled": True,
                    "size": 2,  # 2-way pipeline parallelism
                    "micro_batches": args.gradient_accumulation_steps
                }
            }
        )
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision="fp16",
            deepspeed_plugin=plugin
        )
    elif args.distributed_type == "fsdp":
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_type=StateDictType.FULL_STATE_DICT,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            param_init_fn=None,
            cpu_offload=CPUOffload(offload_params=args.fsdp_offload),
            mixed_precision_policy=MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float32
            ),
            use_orig_params=True,
            sharding_strategy=ShardingStrategy.FULL_SHARD
        )
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision="fp16",
            fsdp_plugin=fsdp_plugin
        )
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision="fp16"
        )
    
    # 데이터셋 로드 (캐시 경로 지정)
    dataset = load_dataset(
        "glue",
        "sst2",
        cache_dir=os.path.join(args.dataset_dir, "glue")
    )
    
    # 토크나이저 로드 (캐시 경로 지정)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=os.path.join(args.cache_dir, "tokenizer")
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )
        outputs["labels"] = examples["label"]  # label을 labels로 복사
        return outputs
    
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,  # 원본 컬럼 모두 제거
    )
    
    # 데이터로더 생성
    def collate_fn(examples):
        return {
            key: torch.tensor([example[key] for example in examples])
            for key in examples[0].keys()
        }
    
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )
    
    # 모델 초기화 (캐시 경로 지정)
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=os.path.join(args.cache_dir, "config")
    )
    config.num_labels = 2
    config.use_cache = False  # gradient checkpointing을 위해 필요
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=os.path.join(args.cache_dir, "model")
    )
    
    # 모든 파라미터를 float32로 변환
    model = model.to(torch.float32)
    
    # Gradient Checkpointing 활성화
    model.gradient_checkpointing_enable()
    
    # PEFT 설정 적용
    peft_config = get_peft_config(args.peft_type, args)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 옵티마이저 및 스케줄러 설정
    if args.distributed_type == "deepspeed":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
    else:
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            optim_bits=8,
            min_8bit_size=4096
        )
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(max_train_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )
    
    # Accelerator로 모든 객체 준비
    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, scheduler
    )
    
    # 평가 메트릭 초기화
    metric = evaluate.load("glue", "sst2")
    
    # 학습 루프
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                
                loss = outputs.loss
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.detach().float()
            
            if step % 100 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch: {epoch}, Step: {step}, Average Loss: {avg_loss}")
        
        # 평가
        model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            metric.add_batch(predictions=predictions, references=references)
        
        eval_metric = metric.compute()
        print(f"Epoch {epoch}: {eval_metric}")
        
        # 모델 저장
        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            output_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            unwrapped_model.save_pretrained(
                output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main() 