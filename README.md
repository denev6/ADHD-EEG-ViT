# Children ADHD detection with EEG signal

## Inference

```bash
pip install -r requirements.txt
python inference.py --dataset "eeg.pt" --fp16
```

The dataset is expected to be a dictionary containing 'data' and 'label' keys. For implementation specifics, please check [EEGDataset](/utils/data.py).

## Model

- Trained model: ["ieee-transformer_250303001232982598_3.pt"](/log)
- Conv1d embedding(ViT) + Transformer(EEG-Transformer)
- Instead of using CLS token(ViT), data classification leverages the entire set of vectors(EEG-Transformer).
- Implementation: [transformer.py](/models/transformer.py)

| Accuracy | Recall | F1-score |
|:--------:|:------:|:--------:|
|  0.972   | 0.952  |  0.976   |

### Reference

- `EEG-Transformer`: Y. He et al., “Classification of attention deficit/hyperactivity disorder based on EEG signals using a EEG-Transformer model,” _J. Neural Eng._, vol. 20, no. 5, Sep. 2023.
- `ViT`: A. Dosovitskiy et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” _arXiv preprint arXiv:2010.11929_, 2021.

## Dataset

- "_EEG Data ADHD-Control Children_" from IEEE dataport (CC BY 4.0).
- Data consists of 19-channel EEG signals, classified into two categories: Control and ADHD.
- "[/assets](/assets)" for more information.

## Blog

[EEG 신호를 활용한 청소년 ADHD 진단](https://denev6.github.io/projects/2025/03/05/eeg-transformer.html)
