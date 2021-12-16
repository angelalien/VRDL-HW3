# VRDL_HW3

Code for Selected Topics in Visual Recognition using Deep Learning(2021 Autumn NYCU) Homework3: Nuclei segmentation. 

## Requirements

I use Google Colab as the environment.
Tho install and import the required library and tools, run the command on Google Colab:

```setup
import os
import shutil
import cv2
import numpy as np
import json
import torch
from os import listdir
from os.path import isfile, isdir, join
from google_drive_downloader import GoogleDriveDownloader as gdd

!pip install pyyaml==5.1

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/$CUDA_VERSION/torch$TORCH_VERSION/index.html

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
setup_logger()
```
> This command has been included in train.ipynb file.
## Dataset Preparation

To download the training dataset, run this command in the Google Colab:
```
gdd.download_file_from_google_drive(file_id='1nEJ7NTtHcCHNQqUXaoPk55VH3Uwh4QGG',
                                    dest_path='./dataset.zip',
                                    unzip=True)
```
> This command has been included in train.ipynb file.

## Training 

To train the model, run the train.ipynb file on Google Colab.

> This command has been included in train.ipynb file.

After running, you will get some output files: nucleus_cocoformat.json, model_final.pth

## Pre-trained Models

You can download pretrained models here:

- [My ResNeXt model](https://drive.google.com/file/d/12j_E_J-j2RSC0hGzapnNp2oZ87IisneI/view?usp=sharing) trained on given nuclei dataset.
  

Model's hyperparameter setting:

-  DATALOADER.NUM_WORKERS = 2
-  SOLVER.IMS_PER_BATCH = 2
-  SOLVER.BASE_LR = 0.00025
-  SOLVER.MAX_ITER = 3000
-  MODEL.ROI_HEADS.NUM_CLASSES = 1
-  MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE=128
-  MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5



## Make Submission

To make the submission file, run the [inference.ipynb](https://colab.research.google.com/drive/1SPr4QLTe1hZbO7Qux7goLgRWOC-RrBuK?usp=sharing).
After running, you will get an output file: answer.json for submission

## Result

My model achieves the following performance on CodaLab:
| Model name  | Top 1 mAP    |
| ----------- |------------- |
| My model  |    24.237%   |
