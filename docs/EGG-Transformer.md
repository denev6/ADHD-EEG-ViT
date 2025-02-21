# Classification of attention deficit/hyperactivity disorder based on EEG signals using a EEG-Transformer model

## Introduction

1. EEG-Transformer는 EEG 신호 분류를 진행함. EEG 신호 특징을 추출해 사용
2. 내부 모듈의 기능과 성능을 분석함
3. 모델 구조를 조정하며 최고 분류 성능 달성해 모델 수정을 위한 이론적 기반을 제공

## Method

### Structure of the transformer model

- Positional Embedding
- Attention modules (x N)
  - MultiHead Attention
  - Add & Norm
  - Feed Forward (Dense + ReLU + Dense)
  - Add & Norm
- Gloabal max pooling
- Fully connection
- Softmax

### Positional embedding layer

학습 가능한 positional embedding 사용

### Multi-head attention layer

Multi-head: 기본 self-attetion을 concat

### Feed Forward layer

2개 fully connected layer + ReLU

### Add & Norm

residual network + normalization

$Norm(X + Attention)$

### Global max pooling

차원 축소를 담당

### Model training process

- epoch: 300
- batch: 256
- optimizer: Adam
- learning-rate: 0.001
- check point + Early Stopping + LR scheduler
- Early Stopping patience: 30 epochs

### Performance metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC curve
- AUC value

## Experimental results

- 30 epoch에서 빠르게 수렴
- Attention과 Add & Norm이 매우 중요함
- Transformer block 2개보다 4개가 좋음. 4, 6, 8은 비슷
- Attention Head는 2개보다 4개가 좋음. 4개보다 6개가 살짝 더 좋음. 8개는 비슷함.
