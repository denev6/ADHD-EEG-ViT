# About Logs

## tag

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
- Detection of ADHD from EEG signals using new hybrid decomposition and deep learning techniques, 2023.

## models

### Transformer

- Reference: Detection of ADHD from EEG signals using new hybrid decomposition and deep learning techniques, 2023.

### ViTransformer

- Transformer(above) + Conv1d embedding
- Instead of a CLS token, data classification leverages the entire set of vectors.
- Reference: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, 2021.
