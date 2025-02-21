# Thought2Text: Text Generation from EEG Signal using Large Language Models (LLMs)

## Introduction

LLM을 활용해 EEG 신호를 텍스트로 변환

1. 시각 자극을 통해 언어에 구애받지 않는 뇌파 신호 캡쳐
2. Multi-channel encoder를 통해 신호를 임베딩
3. 언어 모델 튜닝을 통해 이미지와 EEG 임베딩

추론 과정에는 EEG 신호와 텍스트만 입력으로 사용

- 128 채널 EEG 데이터 사용
- (이미지) 설명은 GPT-4Omni로 생성 & 사람이 검수
- Fine-tune LLM으로 MISTRAL-V3, LLAMA-V3, QWEN2.5를 사용

## Dataset and the need for Visual Stimuli

이미지 인식은 직관적인데 비해 언어 인식은 읽기, 해석을 포함해 복잡한 활동이 일어남. 또한 open-vocabulary 해석에서는 데이터 양이 너무 많아 비효율적임. 따라서 주요한 문제는

- a. 언어 처리의 간섭을 최소화하면서 언어에 구애받지 않는 신경 신호를 활용
- b. 신호를 사용하여 목표 언어로 텍스트를 생성 (특정 명령어나 프롬프트를 지정하여 수행)

이러한 이유로 텍스트가 아닌 시각 정보를 기록한 EEG 신호를 선택

- a에 대해, 이미지를 사용함으로써 언어 모델의 복잡도를 줄임. 눈에 띄는 이미지 특징에 대한 뇌 반응을 이끌어내어 언어에 구애받지 않는 방식으로 신경 활동을 포착하는 데 더 적합
- b에 대해, GPT-4 Omni 같은 이미지 캡션 생성 툴을 이용해 <eeg, text, image> 튜플을 생성

CVPR2017 데이터셋 사용

- 40개 카테고리에 대한 50개 이미지
- 128 채널을 1kHz로 0.5초 동안 샘플링
- second-order Butterworth bandpass filter로 전처리 (5Hz ~ 95Hz 사이)
- 50Hz에 notch filter 사용
- 첫 20개 샘플(20ms)은 지움 -> 이전 이미지와의 간섭 고려
- 440 샘플 길이로 정규화 (N < 500)
- 55-95Hz 범위의 신호만 샘플링 (선행 연구 참고)

CVPR2017은 이미지에 대한 텍스트 설명이 부족하기 때문에 GPT-4를 이용해 생성. 생성한 텍스트는 사람에 의해 fluency와 adequacy를 평가

## Method

### Stage1: Training EEG Encoder for Embedding Extraction

Encoder는 신호를 임베딩($H_{eeg}$)으로 변환.

- 사전학습된 이미지 인코더로부터 파생한 인코더 사용
- 가장 두드러지는 객체를 임베딩

ChannelNet에서 영감받은 CNN 기반 인코더
임베딩은 MLP classifier를 통해 레이블($y_{obj}$)로 분류

Loss는 객체 예측에 대한 Cross-entropy loss와 EEG 임베딩에 대한 Mean Squared loss 평균으로 계산

이미지는 단순화를 위해 Gaussian blur를 적용하고 Canny filter를 이용해 스케치로 변환(edge만 남김)

$H_{egg}$와 $y_{obj}$를 모두 예측하는데 3가지 이유가 있음

1. 이후 멀티모달 비전-언어 모델로 확장하기 위해
2. 임베딩이 이미지 객체에 대해 잘 진행되었는다는 것을 확인하기 위해
3. 객체 레이블이 이후 멀티모달에서 더 정확한 결과를 도출할 수 있기 때문

### Stage2: Priming LLMs with Image Embeddings

projector는 비전과 EEG 신호를 LLM 토큰 임베딩으로 변환
간단한 feed-forward layer로 구성

```json
 {"role": "system", "content": "You are a helpful assistant."},
 {"role": "user", "content": "<image> <object_string> Describe this in one sentence:"},
```
{ Input Prompt }
