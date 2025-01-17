import os
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from peft import (
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptEncoderConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from evaluate import load
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import nltk
nltk.download('punkt')

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="google/gemma-2-27b-it",
        metadata={
            "help": "사용할 모델 선택 (Gemma 9B/27B 또는 Qwen 72B)",
            "choices": [
                "google/gemma-2-2b-it",
                "google/gemma-2-9b-it",
                "google/gemma-2-27b-it",
                "Qwen/Qwen2.5-72B-Instruct"
            ]
        }
    )
    cache_dir: Optional[str] = field(
        default="./cache",
        metadata={"help": "모델 캐시 디렉토리"}
    )
    hf_token: Optional[str] = field(
        default=None,
        metadata={"help": "Hugging Face 토큰 (Gemma 모델 접근용)"}
    )
    liger_kernel: bool = field(
        default=True,
        metadata={"help": "Liger Kernel 사용 여부"}
    )
    flash_attention: bool = field(
        default=True,
        metadata={"help": "Flash Attention 사용 여부"}
    )

@dataclass
class DataTrainingArguments:
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "학습 데이터 수 제한"}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "평가 데이터 수 제한"}
    )
    max_length: Optional[int] = field(
        default=512,
        metadata={"help": "최대 시퀀스 길이"}
    )
    dataset_dir: Optional[str] = field(
        default="./dataset",
        metadata={"help": "데이터셋 캐시 디렉토리"}
    )

@dataclass
class PeftArguments:
    peft_type: str = field(
        default="lora",
        metadata={"help": "PEFT 방식 선택 (lora, prefix, prompt, p-tuning)"}
    )
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)
    lora_target_modules: str = field(default="q_proj,v_proj")
    num_virtual_tokens: int = field(default=20)
    encoder_hidden_size: int = field(default=512)

@dataclass
class QuantizationArguments:
    quantization: str = field(
        default="none",
        metadata={"help": "양자화 방식 (none, 4bit, 8bit)"}
    )
    double_quant: bool = field(
        default=False,
        metadata={"help": "Double Quantization 사용"}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "4비트 양자화 타입 (nf4, fp4)"}
    )

@dataclass
class EvalArguments:
    eval_metrics: str = field(
        default="rouge,bleu,meteor",
        metadata={"help": "평가에 사용할 메트릭 (콤마로 구분)"}
    )
    do_sample: bool = field(
        default=True,
        metadata={"help": "생성 시 샘플링 사용"}
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "생성 시 온도"}
    )
    top_p: float = field(
        default=0.9,
        metadata={"help": "생성 시 상위 확률 합"}
    )
    max_new_tokens: int = field(
        default=128,
        metadata={"help": "생성할 최대 토큰 수"}
    )

def main():
    # 로깅 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # 경고 필터 설정
    import warnings
    warnings.filterwarnings("ignore", message=".*cache_implementation.*")
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # 분산 학습 환경 변수 설정
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    print("local_rank : ", local_rank)
    if local_rank != -1:
        os.environ["RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", str(torch.cuda.device_count()))
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    
    parser = HfArgumentParser((
        ModelArguments,
        DataTrainingArguments,
        PeftArguments,
        QuantizationArguments,
        EvalArguments,
        TrainingArguments,
    ))
    
    model_args, data_args, peft_args, quant_args, eval_args, training_args = parser.parse_args_into_dataclasses()
    
    if local_rank == -1:
        # 단일 GPU 또는 CPU 모드
        training_args.local_rank = local_rank
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        # 분산 학습 모드
        training_args.local_rank = local_rank
        training_args.ddp_backend = "nccl"
        training_args.parallel_mode = "distributed"
        device = torch.device(f"cuda:{local_rank}")
    
    # Hugging Face 토큰 설정
    if model_args.hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = model_args.hf_token
    else:
        raise ValueError(
            "Gemma 모델에 접근하기 위해서는 Hugging Face 토큰이 필요합니다. "
            "--hf_token 인자로 토큰을 전달하세요."
        )
    
    # 시드 설정
    set_seed(training_args.seed)
    
    # 데이터셋 로드
    dataset = load_dataset(
        "tatsu-lab/alpaca",
        cache_dir=os.path.join(data_args.dataset_dir, "alpaca")
    )
    
    # 학습/검증 세트로 분할
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=training_args.seed)
    dataset = {
        "train": dataset["train"],
        "validation": dataset["test"]
    }
    
    if data_args.max_train_samples is not None:
        dataset["train"] = dataset["train"].select(range(min(len(dataset["train"]), data_args.max_train_samples)))
    if data_args.max_eval_samples is not None:
        dataset["validation"] = dataset["validation"].select(range(min(len(dataset["validation"]), data_args.max_eval_samples)))
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        padding_side="right",
        use_fast=False,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
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
        
        tokenized = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=data_args.max_length,
            return_tensors=None,
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # 토크나이징 적용
    tokenized_datasets = {
        split: dataset[split].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset[split].column_names,
        )
        for split in dataset.keys()
    }
    
    # 양자화 설정
    if quant_args.quantization != "none":
        compute_dtype = torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=quant_args.quantization == "4bit",
            load_in_8bit=quant_args.quantization == "8bit",
            bnb_4bit_quant_type=quant_args.quant_type,
            bnb_4bit_double_quant=quant_args.double_quant,
            bnb_4bit_compute_dtype=compute_dtype
        )
    else:
        quant_config = None
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        quantization_config=quant_config,
        torch_dtype=torch.float16 if quant_args.quantization != "none" else torch.float32,
        attn_implementation="flash_attention_2" if model_args.flash_attention else "eager",
        use_cache=None,  # use_cache를 None으로 설정
    )
    
    if quant_args.quantization != "none":
        model = prepare_model_for_kbit_training(model)
    
    # PEFT 설정
    if peft_args.peft_type == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=peft_args.lora_r,
            lora_alpha=peft_args.lora_alpha,
            lora_dropout=peft_args.lora_dropout,
            target_modules=peft_args.lora_target_modules.split(","),
            bias="none",
        )
    elif peft_args.peft_type == "prefix":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
            prefix_projection=True,
            encoder_hidden_size=peft_args.encoder_hidden_size,
        )
    elif peft_args.peft_type == "prompt":
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
        )
    elif peft_args.peft_type == "p-tuning":
        peft_config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
            encoder_hidden_size=peft_args.encoder_hidden_size,
        )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 모델을 해당 GPU로 이동
    model = model.to(device)
    
    # 분산 학습 모드에서 DistributedDataParallel 래핑
    if training_args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[training_args.local_rank],
            output_device=training_args.local_rank,
            find_unused_parameters=False
        )
    
    class CustomTrainer(Trainer):
        def compute_metrics(self, eval_preds):
            metrics = {}
            predictions, labels = eval_preds
            
            # 토큰 ID를 텍스트로 디코딩
            predictions = self.tokenizer.batch_decode(
                predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            labels = self.tokenizer.batch_decode(
                labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            
            # 평가 메트릭 계산
            metric_names = eval_args.eval_metrics.split(",")
            
            if "rouge" in metric_names:
                rouge = load("rouge")
                rouge_output = rouge.compute(
                    predictions=predictions,
                    references=labels,
                    use_aggregator=True
                )
                metrics.update({k: v for k, v in rouge_output.items()})
            
            if "bleu" in metric_names:
                bleu_scores = []
                for pred, label in zip(predictions, labels):
                    pred_tokens = nltk.word_tokenize(pred)
                    label_tokens = nltk.word_tokenize(label)
                    bleu_score = sentence_bleu([label_tokens], pred_tokens)
                    bleu_scores.append(bleu_score)
                metrics["bleu"] = np.mean(bleu_scores)
            
            if "meteor" in metric_names:
                meteor = load("meteor")
                meteor_output = meteor.compute(
                    predictions=predictions,
                    references=labels
                )
                metrics["meteor"] = meteor_output["meteor"]
            
            return metrics
        
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            if not prediction_loss_only:
                gen_kwargs = {
                    "max_new_tokens": eval_args.max_new_tokens,
                    "do_sample": eval_args.do_sample,
                    "temperature": eval_args.temperature,
                    "top_p": eval_args.top_p,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                generated_tokens = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_kwargs
                )
                return (None, generated_tokens, inputs["labels"])
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    # Trainer 초기화
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )
    
    # Liger Kernel 활성화
    if model_args.liger_kernel:
        trainer.accelerator.state.use_liger_kernel = True
    
    # 학습 실행
    train_result = trainer.train()
    
    # 최종 모델 저장
    trainer.save_model()
    trainer.save_state()
    
    # 학습 결과 출력
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # 평가 실행
    eval_results = trainer.evaluate()
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)

if __name__ == "__main__":
    main() 