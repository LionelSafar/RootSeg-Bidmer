# rootseg

RootSeg provides an end-to-end pipeline for root image processing, including reformatting of the files, image preprocessing, semantic segmentation and post-processing for binary root classification and multiclass-segmentation for multiple species or also functional groups to evaluate root-dynamics of time-series of minirhizotron images.

This Repository resulted from long-term observation of alpine grasslands on the Bidmer plateau and Furkapass in the Swiss Alps, in order to automate the image analysis procedure. The NN pipeline can be applied to differen root-datasets as well.

---

## Features

- Preprocessing pipelines for root image reformatting, normalization and filtering, semi-manual masking of non-root regions 
- Deep learning based root semantic segmentation models (e.g., UNet variants)  
- Visualization functions for multiclass segmentation results  
- Support for running optimization and hyperparameter tuning  
- Tools for root change detection and analysis  

---

## Installation

You can install `rootseg` directly from GitHub:

```bash
pip install git+https://github.com/LionelSafar/RootSeg-Bidmer.git
```

## Project Status

Detailled description as well as artificial data generation and post-processing modules will follow..
