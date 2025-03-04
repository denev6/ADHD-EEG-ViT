# Dataset: IEEE ADHD-control-children

Download: [eeg-data-adhd-control-children](https://ieee-dataport.org/open-access/eeg-data-adhd-control-children)

> Since one of the deficits in ADHD children is visual attention, the EEG recording protocol was based on visual attention tasks. In the task, a set of pictures of cartoon characters was shown to the children and they were asked to count the characters. The number of characters in each image was randomly selected between 5 and 16, and the size of the pictures was large enough to be easily visible and countable by children. To have a continuous stimulus during the signal recording, each image was displayed immediately and uninterrupted after the child’s response. Thus, the duration of EEG recording throughout this cognitive visual task was dependent on the child’s performance (i.e. response speed).
>
> IEEE data port

![v1p](/assets/signal.png)

## Data Shape

- **Samples:** 61 ADHD, 60 Control  
- **Sequence Length:** 7,938 ~ 43,252  
- **Channels:** 19  

## Preprocessing

- Segmented into fixed-length subsets of **9,250**
  - [preprocess.ipynb](/assets/tools/preprocess.ipynb) for details
- Reference: M. Y. Esas and F. Latifoglu, “Detection of ADHD from EEG signals using new hybrid decomposition and deep learning techniques,” _Journal of Neural Engineering_, vol. 20, no. 3, Jun. 2023.

### Case 1: Fixed

- **Train:** 70% → Shape: `[121, 19, 9250]`  
- **Validation:** 10% → Shape: `[17, 19, 9250]`  
- **Test:** 20% → Shape: `[36, 19, 9250]`

### Case2: Cross Validation

- **Train:** 80% → Shape: `[138, 19, 9250]`  
  - **Validation:** 1/5 of training set
- **Test:** 20% → Shape: `[36, 19, 9250]`
