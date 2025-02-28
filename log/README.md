# Logs

## Tag

### IEEE_22

- Segmented into fixed-length subsets of **2,560** (20ms at 128Hz) 
  - **Train:** 70% → Shape: `[547, 19, 2560]`  
  - **Validation:** 10% → Shape: `[77, 19, 2560]`  
  - **Test:** 20% → Shape: `[159, 19, 2560]`
- EPSPatNet86: eight-pointed star pattern learning network for detection ADHD disorder using EEG signals, 2022.

### IEEE_23

- Segmented into fixed-length subsets of **9,250**
  - **Train:** 70% → Shape: `[121, 19, 9250]`  
  - **Validation:** 10% → Shape: `[17, 19, 9250]`  
  - **Test:** 20% → Shape: `[36, 19, 9250]` 
-  Detection of ADHD from EEG signals using new hybrid decomposition and deep learning techniques, 2023.

## Models

### EEG-Transformer + ViT

- Trained model: "ieee-transformer_250228075729819009"
- `EEG-Transformer`: Classification of attention deficit/hyperactivity disorder based on EEG signals using a EEG-Transformer model, 2023.
- `ViT`: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, 2021.
- Transformer(EEG-Transformer) + Conv1d embedding(ViT)
- Instead of a CLS token(ViT), data classification leverages the entire set of vectors(EEG-Transformer).
