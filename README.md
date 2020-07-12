# AMTA-Net
## Asymmetric Multi-Task Attention Network for Prostate Bed Segmentation in CT Images.
This is a python (PyTorch) implementation of the **Asymmetric Multi-Task Attention Network** for prostate bed segmentation in CT images proposed in our paper [**Asymmetrical Multi-Task Attention U-Net for the Segmentation of Prostate Bed in CT Image**] accepted by **MICCAI 2020**.

<img src="./fig1.png"/>
<img src="./fig2.png"/>

This code has been tested and passed on `Ubuntu 16.04`.

## How to use
### Prepare data
The dataset folder is organized in the following way:

```
└── primary directory
    ├── IMG_slice # sub folder storing image slice files
    |   ├── case001.nii.gz
    |   |   ├── slice001.nii.gz
    |   |   |   ...
    |   |   └── sliceXXX.nii.gz
    |   |   ...
    |   └── caseXXX.nii.gz
    |       ├── slice001.nii.gz
    |       |   ...
    |       └── sliceXXX.nii.gz
    ├── PB_slice # sub folder storing prostate bed mask slice files
    |   ├── case001.nii.gz
    |   |   ├── slice001.nii.gz
    |   |   |   ...
    |   |   └── sliceXXX.nii.gz
    |   |   ...
    |   └── caseXXX.nii.gz
    |       ├── slice001.nii.gz
    |       |   ...
    |       └── sliceXXX.nii.gz
    ├── OAR_slice # sub folder storing OAR (i.e., bladder and rectum) mask slice files
    |   ├── case001.nii.gz
    |   |   ├── slice001.nii.gz
    |   |   |   ...
    |   |   └── sliceXXX.nii.gz
    |   |   ...
    |   └── caseXXX.nii.gz
    |       ├── slice001.nii.gz
    |       |   ...
    |       └── sliceXXX.nii.gz
    ├── PB # sub folder storing prostate bed mask volume files
    |   ├── case001.nii.gz
    |   |   ...
    |   └── caseXXX.nii.gz
    ├── OAR # sub folder storing OAR (i.e., bladder and rectum) mask volume files
    |   ├── case001.nii.gz
    |   |   ...
    |   └── caseXXX.nii.gz
    └── trained_models # output folder storing trained model files and predicted masks
```

## Citation

Please cite our paper if it is useful for your research: