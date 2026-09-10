# rootseg

RootSeg provides an end-to-end pipeline for root image processing, including reformatting of the files, image preprocessing, semantic segmentation and post-processing for binary root classification and multiclass-segmentation for functional groups to evaluate root-dynamics of time-series of minirhizotron images.

This Repository resulted from long-term observation of alpine grasslands on the Bidmer plateau and Furkapass in the Swiss Alps in collaboration with ALPFOR, in order to automate the image analysis procedure. The NN pipeline can be applied to different root-datasets as well, however there are data-specific adaptations made for the pipeline.

---

## Features

- Preprocessing pipelines for root image reformatting, normalisation and filtering, semi-manual masking of non-root regions 
- Training pipelines for deep learning based root semantic segmentation models
- Inference pipeline to segment preprocessed root images  
- Visualisation functions for multiclass segmentation results  
- Support for running optimisation for hyperparameter tuning  


## Installation

You can install `rootseg` directly from GitHub:

```bash
pip install git+https://github.com/LionelSafar/RootSeg-Bidmer.git
```


## Setup of the folder system

Run `rootseg.preprocess.rename_and_order` on a folder of images containing a set of root scans (in `.tiff` format), renames the images to `{experiment_name}_T{i}_L{j}_YYYY.mm.dd_{aux}.tiff`, where a folder system `Raw_images` is created where all images are sorted according to their tube.

The data folder system will be structured as follow:

```
Root-Segmentation/
└── Data/
    └── {experiment_name}/
        └── Raw_images/
            └── T{i}/
```

To work properly, all images are assumed to contain the following substrings to reformat the data correctly:
- Levels: `_L{i}_`, where $i \in \mathbb{N}$ (e.g. "L1", "L2"...) - L1 is assumed to be the topmost image of the series, possibly containing a tape to be removed
- Tubes: `_T{j}_`, where $j \in \mathbb{N}$ (e.g. "T1", "T2"...)
- Date: `YYYY.mm.dd` or `dd.mm.YYYY` or `dd.mm.yy` - Everything after the date is considered auxiliary information, ensure tube, level and orientation are specified before the date

## Preprocess the raw images

The `rootseg.preprocess.preprocessor` script takes a raw folder and allows to:
- stitch images (using phasecorrelation) if multiple levels occur at each specific scanning date
- removes specific regions (e.g. taped regions), this requires a mask folder containing masks for each tube
- align timeseries in the tube, which aligns images from a localized region at the bottom
- remove scannoise
- adjust histogram
- adjust brightness and contrast to a fixed reference value


The folder structure afterwards will contain:

```
Root-Segmentation/
└── Data/
    └── {experiment_name}/
        └── masks/
        └── preprocessed/
        └── preview/
        └── Raw_images/
```

where the `masks` folder has to be provided beforehand, containing tube-specific masks in RGB, where non-masked regions should be set to black. 

## segment the preprocessed images

The `rootseg.inference.segment` script takes a preprocessed folder and a trained binary segmenter `.pth` file as well as a multiclass segmenter in the multiclassification case (optionally the multiclass case allows to provide binary images instead of a binary segmenter as well). In the binary case a single segmentation folder is generated, in the multiclass case the following foldersystem is generated:

```
Root-Segmentation/
└── .../
    └── segmentation/
        └── output/
        └── predmap/
        └── prob_predmap/
        └── rgb/
```

where `output/` contains the direct segmented images with class labels 0,1,2, ..
`predmap` is the classification map without rootmask overlay (argmax of `prob_predmap`)
`prob_predmap` shows the probability prediction of each of the three classes (softmax) and maximum confidence of the most likely class
`rgb` contains the visual output of `output` where B=Carex, G=herb, R=graminoid.

In the multiclass case, `rootseg.inference.multiclass_to_binary` transforms multiclass maps to binary maps for each class.

## preparation for Rhizovision

In the case of splitting the images to specific depths use `rootseg.inference.split_depths` on the segmented images by providing a `depth_masks` folder, containing the tube specific depth masks. This generates a new split_depths folder containing all split images and an `areas.txt` file containing the area (in cm$^2$) of each image as table considering masked areas.

In case of no splitting, use `rootseg.inference.flatten_segmentations`, which achieves the same structure and produces `areas.txt` without splitting.

## Training a model

A model can be trained with `rootseg.training.train_model` by selecting a model and classification type. It requires a datafolder with:


```
Data/
└── train/
    └── annotations/
    └── images/
└── val/
    └── annotations/
    └── images/
└── transfer/
    └── annotations/
    └── images/
```
where `transfer` is only mandatory for transfer learning.

the images and annotations need to be named the same as the images are only matched via alphabetical sorting.
In the transfer learning case, images from `transfer` images will be inflated by 50% by images from `train` for the training dataset.
