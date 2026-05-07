# ZScribbleSeg: A Comprehensive Segmentation Framework with Efficient Scribble Supervision and Prior-Guided Correction

This repository provides the official implementation of our Medical Image Analysis paper:

**ZScribbleSeg: A Comprehensive Segmentation Framework with Modeling of Efficient Annotation and Maximization of Scribble Supervision**


ZScribbleSeg is a comprehensive framework for scribble-supervised medical image segmentation. We first investigate the principle of good scribble annotations, which leads to efficient scribble forms via supervision maximization and randomness simulation. We further introduce regularization terms to encode the spatial relationship and the shape constraints, where the EM algorithm is utilized to estimate the mixture ratios of label classes. These ratios are critical in identifying the unlabeled pixels for each class and correcting erroneous predictions, thus the accurate estimation lays the foundation for the incorporation of spatial prior.


<img width="3096" height="1412" alt="pipeline" src="https://github.com/user-attachments/assets/e1914fa9-5e4a-4de2-9afc-3db55d0f6008" />


# Datasets
1. The MSCMR dataset with mask annotations can be downloaded from [MSCMRseg](https://zmiclab.github.io/zxh/0/mscmrseg19/data.html).
2. Our scribble annotations of MSCMRseg have been released in [MSCMR_scribbles](https://github.com/BWGZK/CycleMix/tree/main/MSCMR_scribbles). Please cite this paper if you use the scribbles for your research.
3. The scribble-annotated MSCMR dataset used for training could be directly downloaded from [MSCMR_dataset](https://github.com/BWGZK/CycleMix/tree/main/MSCMR_dataset). 
4. The ACDC dataset with mask annotations can be downloaded from [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/) and the scribble annotations could be downloaded from [ACDC scribbles](https://vios-s.github.io/multiscale-adversarial-attention-gates/data). Please organize the dataset as the following structure:
```
XXX_dataset/
  -- TestSet/
      --images/
      --labels/
  -- train/
      --images/
      --labels/
  -- val/
      --images/
      --labels/
```


# Usage

## 1. Environment Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

This code has been tested with:

```text
Python 3.x
PyTorch
torchvision
numpy
scipy
SimpleITK
nibabel
opencv-python
```

Please refer to `requirements.txt` for detailed package versions.

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset ACDC \
    --data_root /path/to/ACDC_dataset \
    --output_dir ./checkpoints/ACDC_ZScribbleSeg
```

Please modify the dataset name, data path, GPU settings, and training parameters according to your environment.


# Main Components

ZScribbleSeg mainly consists of the following components:

- Efficient scribble supervision
- Supervision maximization
- Randomness simulation
- EM-based class mixture ratio estimation
- Spatial prior modeling
- Shape-constrained correction
- Unified scribble-supervised segmentation framework



# Citation

Please cite our paper if you use ZScribbleSeg or the related scribble annotations in your research.

```bibtex
@article{ZHANG2026104074,
title = {ZScribbleSeg: A comprehensive segmentation framework with modeling of efficient annotation and maximization of scribble supervision},
journal = {Medical Image Analysis},
volume = {112},
pages = {104074},
year = {2026},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2026.104074}
```


# Contact

If you have any questions, please feel free to open an issue or contact us.

Thanks for your attention.
