# AMTA-U-Net
This is a python (PyTorch) implementation of the **Asymmetric Multi-Task Attention U-Net** for prostate bed segmentation in CT images. This method is proposed in our paper [**Asymmetrical Multi-Task Attention U-Net for the Segmentation of Prostate Bed in CT Image**] accepted by **MICCAI 2020**.

<img src="./fig1.png"/>

This code has been tested and passed on `Ubuntu 16.04`.

## Citation

Please cite our paper if it is useful for your research:

    @inproceedings{xu2020amtanet, 
      title = {Asymmetrical Multi-Task Attention U-Net for the Segmentation of Prostate Bed in CT Image},
      author = {Xu, Xuanang and Lian, Chunfeng and Wang, Shuai and Wang, Andrew and Royce, Trevor and Chen, Ronald and Lian, Jun and Shen, Dinggang},
      booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
      year = {2020},
      organization={Springer}
    }

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