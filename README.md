# Gemma 모델 파인튜닝

이 프로젝트는 Google의 Gemma 모델을 다양한 PEFT(Parameter-Efficient Fine-Tuning) 방법을 사용하여 파인튜닝하는 코드를 제공합니다.

## 지원하는 모델
- google/gemma-2-27b-it
- google/gemma-2-7b-it
- google/gemma-2b-it

## 지원하는 PEFT 방법
- LoRA (Low-Rank Adaptation)
- Prefix Tuning
- Prompt Tuning
- P-Tuning v2

## 설치 방법

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# 필요한 패키지 설치
pip install -r requirements.txt
```

## 사용 방법

### 기본 학습 명령어

```bash
python train.py \
    --model_name_or_path google/gemma-2-27b-it \
    --peft_type lora \
    --num_epochs 3 \
    --batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 16
```

### PEFT 방법별 설정

#### 1. LoRA
```bash
python train.py \
    --peft_type lora \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_target_modules "q_proj,v_proj"
```

#### 2. Prefix Tuning
```bash
python train.py \
    --peft_type prefix \
    --num_virtual_tokens 20
```

#### 3. Prompt Tuning
```bash
python train.py \
    --peft_type prompt \
    --num_virtual_tokens 20
```

#### 4. P-Tuning v2
```bash
python train.py \
    --peft_type p-tuning \
    --num_virtual_tokens 20 \
    --encoder_hidden_size 512
```

## DeepSpeed ZeRO-3 설정

현재 코드는 DeepSpeed ZeRO-3 최적화를 사용하며, 다음과 같은 주요 기능을 포함합니다:
- 파라미터 샤딩
- 옵티마이저 상태 샤딩
- CPU 오프로딩
- FP16 혼합 정밀도 학습

## 주의사항

1. **하드웨어 요구사항**
   - Gemma 27B: 최소 2-4개의 고성능 GPU (예: A100 80GB) 권장
   - ZeRO-3와 CPU 오프로딩으로 메모리 사용량 최적화

2. **학습 데이터**
   - 현재 GLUE의 SST-2 데이터셋을 사용
   - 다른 데이터셋으로 변경 시 코드 수정 필요

3. **모델 라이선스**
   - Gemma 모델 사용 시 Google의 라이선스 조건 준수 필요
   - 상업적 사용 시 별도 확인 필요

## 참고 자료
- [Gemma 공식 문서](https://huggingface.co/google/gemma-2-27b-it)
- [PEFT 라이브러리 문서](https://github.com/huggingface/peft)
- [DeepSpeed 문서](https://www.deepspeed.ai/) 