import argparse
import os
import evaluate
import torch
import deepspeed
import time
from tqdm.auto import tqdm
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
    TaskType,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
    AutoConfig,
    BitsAndBytesConfig,
    AutoModelForCausalLM
)
from torch.utils.data import DataLoader
from typing import Dict, List
import bitsandbytes as bnb
import datetime
import logging
import warnings

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
        choices=["no", "deepspeed", "deepspeed_pp", "fsdp"],
        help="분산 학습 방식 선택"
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="CPU 오프로딩 사용 (DeepSpeed 및 FSDP)"
    )
    
    # 분산 학습 관련 인자 추가
    parser.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=2,
        help="Pipeline parallel size"
    )
    parser.add_argument(
        "--pipeline_chunk_size",
        type=int,
        default=1,
        help="Pipeline chunk size"
    )
    
    # Hugging Face 토큰 설정
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face 토큰 (필수: Gemma 모델 접근용)"
    )
    
    # LoRA/QLoRA 관련 인자
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["none", "4bit", "8bit"],
        help="Quantization type for QLoRA"
    )
    parser.add_argument(
        "--double_quant",
        action="store_true",
        help="Use double quantization for QLoRA"
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        default="nf4",
        choices=["nf4", "fp4"],
        help="Quantization data type for QLoRA"
    )
    
    # 메모리 최적화 관련 인자
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing"
    )
    
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="학습 데이터 수 제한 (None인 경우 전체 데이터 사용)"
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="평가 데이터 수 제한 (None인 경우 전체 데이터 사용)"
    )
    
    return parser.parse_args()

def get_peft_config(peft_type: str, args: argparse.Namespace):
    if peft_type == "lora" or peft_type == "qlora":
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
            prefix_projection=True,
            encoder_hidden_size=args.encoder_hidden_size,
            inference_mode=False,
            token_dim=2048,  # Gemma hidden size
            num_attention_heads=8,  # Gemma num attention heads
            num_layers=18  # Gemma num layers
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
    
    # 학습 시작 시간 기록
    start_time = time.time()
    
    # 분산 학습 환경 설정
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"Local Rank: {local_rank}, World Size: {world_size}")
    
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        
        # GPU 장치 정보 설정
        gpu_device_ids = [local_rank]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
        
        init_process_group_kwargs = {
            "backend": "nccl",
            "init_method": "env://",
            "rank": local_rank,
            "world_size": world_size,
            "timeout": datetime.timedelta(minutes=15)
        }
        torch.distributed.init_process_group(**init_process_group_kwargs)
        
        # 프로세스 그룹에 device_ids 설정
        if torch.distributed.group.WORLD is not None:
            torch.distributed.barrier(device_ids=gpu_device_ids)
    
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
    
    # 양자화 설정
    if args.quantization != "none":
        compute_dtype = torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=args.quantization == "4bit",
            load_in_8bit=args.quantization == "8bit",
            bnb_4bit_quant_type=args.quant_type,
            bnb_4bit_double_quant=args.double_quant,
            bnb_4bit_compute_dtype=compute_dtype
        )
    else:
        quant_config = None
    
    # 분산 학습 플러그인 설정
    if args.distributed_type == "deepspeed":
        plugin = DeepSpeedPlugin(
            hf_ds_config={
                "zero_optimization": {
                    "stage": 3,
                    "offload_optimizer": {
                        "device": "cpu" if args.offload else "none",
                        "pin_memory": True if args.offload else False
                    },
                    "offload_param": {
                        "device": "cpu" if args.offload else "none",
                        "pin_memory": True if args.offload else False
                    },
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "stage3_gather_16bit_weights_on_model_save": True,
                    "reduce_bucket_size": int(5e8),
                    "stage3_param_persistence_threshold": int(1e6)
                },
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "gradient_clipping": 1.0,
                "train_micro_batch_size_per_gpu": args.batch_size,
                "train_batch_size": args.batch_size * args.gradient_accumulation_steps * world_size,
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
                    "size": world_size
                },
                "pipeline_parallel": {
                    "enabled": False
                }
            }
        )
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision="fp16",
            deepspeed_plugin=plugin
        )
    elif args.distributed_type == "deepspeed_pp":
        plugin = DeepSpeedPlugin(
            hf_ds_config={
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "gradient_clipping": 1.0,
                "train_micro_batch_size_per_gpu": args.batch_size,
                "train_batch_size": args.batch_size * args.gradient_accumulation_steps * world_size,
                "zero_optimization": {
                    "stage": 1,
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
                    "reduce_bucket_size": int(5e8)
                },
                "fp16": {
                    "enabled": True,
                    "loss_scale": 0,
                    "loss_scale_window": 1000,
                    "initial_scale_power": 16,
                    "hysteresis": 2,
                    "min_loss_scale": 1
                },
                "pipeline": {
                    "enabled": True,
                    "num_stages": args.pipeline_parallel_size,
                    "pipeline_chunk_size": args.pipeline_chunk_size,
                    "activation_checkpoint_interval": 1
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
            cpu_offload=CPUOffload(offload_params=args.offload),
            mixed_precision_policy=MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16
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
    
    # 데이터셋 로드
    raw_datasets = load_dataset(
        "tatsu-lab/alpaca",
        cache_dir=os.path.join(args.dataset_dir, "alpaca")
    )
    
    # 학습/검증 세트로 분할
    raw_datasets = raw_datasets["train"].train_test_split(test_size=0.1, seed=args.seed)
    raw_datasets = {
        "train": raw_datasets["train"],
        "validation": raw_datasets["test"]
    }
    
    if args.max_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(range(min(len(raw_datasets["train"]), args.max_train_samples)))
    if args.max_eval_samples is not None:
        raw_datasets["validation"] = raw_datasets["validation"].select(range(min(len(raw_datasets["validation"]), args.max_eval_samples)))
    
    # 토크나이저 로드 (캐시 경로 지정)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=os.path.join(args.cache_dir, "tokenizer"),
        padding_side="right",
        use_fast=False,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        # Alpaca 형식의 프롬프트 템플릿
        prompts = [
            f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            if input_text.strip()
            else f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            for instruction, input_text, output in zip(
                examples["instruction"], 
                examples["input"], 
                examples["output"]
            )
        ]
        
        # 토크나이징
        tokenized = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors=None,
        )
        
        # 레이블 설정 (다음 토큰 예측을 위해 input_ids를 그대로 사용)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # 토크나이징 적용
    tokenized_datasets = {
        split: dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=raw_datasets[split].column_names,
        )
        for split, dataset in raw_datasets.items()
    }
    
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
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=os.path.join(args.cache_dir, "model"),
        quantization_config=quant_config if args.quantization != "none" else None,
        torch_dtype=torch.float16,  # 모든 파라미터를 float16으로 통일
        low_cpu_mem_usage=True,  # 양자화된 모델을 위해 True로 설정
    )
    
    if args.quantization != "none":
        model = prepare_model_for_kbit_training(model)
        # 양자화된 모델의 모든 파라미터를 float16으로 변환
        for param in model.parameters():
            if param.dtype == torch.uint8:
                param.data = param.data.to(torch.float16)
    
    # Gradient Checkpointing 선택적 활성화
    if args.gradient_checkpointing:
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
    
    # 로깅 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    
    # 경고 필터 설정
    warnings.filterwarnings("ignore", message=".*cache_implementation.*")
    
    # 학습 루프
    total_steps = args.num_epochs * len(train_dataloader)
    progress_bar = tqdm(total=total_steps, desc="Training")
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        epoch_start_time = time.time()
        
        for step, batch in enumerate(train_dataloader):
            step_start_time = time.time()
            
            if args.distributed_type == "deepspeed_pp":
                # Pipeline Parallelism용 처리
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                accelerator.backward(loss)
                
                if step % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                # 일반적인 처리
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
            
            # 진행 상황 업데이트
            step_time = time.time() - step_start_time
            samples_per_second = args.batch_size * args.gradient_accumulation_steps / step_time
            current_lr = optimizer.param_groups[0]["lr"]
            
            if accelerator.is_main_process and step % 10 == 0:
                avg_loss = total_loss / (step + 1)
                progress_bar.set_postfix({
                    'epoch': epoch,
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'samples/s': f'{samples_per_second:.2f}'
                })
                progress_bar.update(1)
        
        # 에폭 종료 시 평가
        model.eval()
        eval_loss = 0
        eval_steps = 0
        
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                eval_loss += outputs.loss.detach().float()
                eval_steps += 1
                
                predictions = outputs.logits.argmax(dim=-1)
                predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
                metric.add_batch(predictions=predictions, references=references)
        
        eval_metric = metric.compute()
        eval_loss = eval_loss / eval_steps
        
        # 에폭 결과 출력
        epoch_time = time.time() - epoch_start_time
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch} Results:")
            print(f"Train Loss: {total_loss / len(train_dataloader):.4f}")
            print(f"Eval Loss: {eval_loss:.4f}")
            print(f"Accuracy: {eval_metric['accuracy']:.4f}")
            print(f"Epoch Time: {epoch_time:.2f}s")
            print(f"Learning Rate: {current_lr:.2e}")
        
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
    
    # 전체 학습 시간 출력
    total_time = time.time() - start_time
    if accelerator.is_main_process:
        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Average epoch time: {total_time/args.num_epochs:.2f}s")

if __name__ == "__main__":
    main() 