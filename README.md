# Children ADHD detection with EEG signal

**Attention Deficit/Hyperactivity Disorder (ADHD)** is one of the most common disorders in children and adolescents. If not properly diagnosed and treated at an early stage, it can have long-term negative effects on academic achievement, social relationships, and emotional development. Among the various brainwave signals that can aid in ADHD diagnosis, **EEG** (Electroencephalogram) is widely used in neuroscience research and clinical diagnosis as a non-invasive method for measuring brain activity.  

This project designed and experimented with a deep learning model to assist in diagnosing adolescent ADHD using EEG signals. Based on the ideas from **Vision Transformer** (A. Dosovitskiy et al., 2021) and **EEG-Transformer** (Y. He et al., 2023), we implemented a transformer-based model. Using the "_EEG Data ADHD-Control Children_" dataset provided by IEEE, we achieved **a high accuracy of 0.972**.  

The key advantages of our model are as follows:  

- End-to-end training is possible without requiring complex preprocessing steps.  
- By utilizing mixed precision techniques, we improved training speed while maintaining high accuracy.  
- The embedding layer was adjusted to enhance scalability, making it easier to apply the model to other EEG datasets.  

However, we observed an overfitting issue during the training process. This is likely due to the limited dataset, and we expect that acquiring additional data or applying data augmentation techniques will improve the model’s robustness in the future.

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

한국어: [EEG 신호를 활용한 청소년 ADHD 진단](https://denev6.github.io/projects/2025/03/05/eeg-transformer.html)
