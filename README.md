# RSNA Intracranial Hemorrhage Detection

The code in this repo demonstrates how to build a baseline deep learning model to detect different types of intracranial hemorrhage from CT scan images. The training data is from a Kaggle competition [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/). I will go through the usual steps applying data science techniques to problem solving, which are exploratory data analysis, model building and training and inferencing.

## Exploratory Data Analysis

As with any data science projects, a large part of effort is in exploratory data analysis (EDA) and data processing. In this particular training dataset, which you can download from the [competition website](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/), there are over 600000 DICOM images. They are all CT scans of the head, of which about one sixth were identified with some sort of hemorrhage. I have divided the EDA into three jupyter-notebooks:

1. [RSNA-Exploratory Data Analysis Part 1](https://github.com/takmanman/RSNA-Intercranial-Hemorrhage-Detection/blob/master/RSNA-Exploratory%20Data%20Analysis%20Part%201.ipynb) In this notebook, we will look at the accompanying .csv file of the training dataset. We will find out what types of hemorrhage images are include and their relative distribution. We will also look for any oddity. In the end we will convert the data into data table that will be used for training the model.

2. [RSNA-Exploratory Data Analysis Part 2](https://github.com/takmanman/RSNA-Intercranial-Hemorrhage-Detection/blob/master/RSNA-Exploratory%20Data%20Analysis%20Part%202.ipynb) In this notebook, we will look at the DICOM images and extract their technical specifications. We will find out if the images have similar technical attributes such as sizes, pixel spacings, samples per pixel, etc. We will put these information in another data table which will be use for preprocessing the images.
3. [RSNA-Exploratory Data Analysis Part 3](https://github.com/takmanman/RSNA-Intercranial-Hemorrhage-Detection/blob/master/RSNA-Exploratory%20Data%20Analysis%20Part%203.ipynb) In this notebook, we will look at the DICOM images with their specified windows. Images from each categories: normal, epidural, intrparenchymal, intraventricular, subarachnoid and subdural will be shown.

## Data Preprocessing



## Model Building and Training

## Inferencing

## Directory Structure
