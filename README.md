# RSNA Intracranial Hemorrhage Detection

The code in this repo demonstrates how to build a baseline deep learning model to detect different types of intracranial hemorrhage from CT scan images. The training data is from the Kaggle competition [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/). I will go through the usual steps of data science problem solving, which are exploratory data analysis, data preprocessing, model building and training and inferencing.

## Exploratory Data Analysis

As with any data science projects, a large part of effort is in exploratory data analysis (EDA) and data processing. In this particular training dataset, which you can download from the [competition website](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/), there are over 600000 DICOM images. They are all CT scans of the head, of which about one sixth were identified with some sort of hemorrhage. I have divided the EDA into three jupyter-notebooks:

1. [RSNA-Exploratory Data Analysis Part 1](https://github.com/takmanman/RSNA-Intercranial-Hemorrhage-Detection/blob/master/RSNA-Exploratory%20Data%20Analysis%20Part%201.ipynb) In this notebook, we will look at the accompanying .csv file of the training dataset. We will find out what types of hemorrhage images are include and their relative distribution. We will also look for any oddity. In the end we will convert the data into data table that will be used for training the model.

2. [RSNA-Exploratory Data Analysis Part 2](https://github.com/takmanman/RSNA-Intercranial-Hemorrhage-Detection/blob/master/RSNA-Exploratory%20Data%20Analysis%20Part%202.ipynb) In this notebook, we will look at the DICOM images and extract their technical specifications. We will find out if the images have similar technical attributes such as sizes, pixel spacings, samples per pixel, etc. We will put these information in another data table which will be use for preprocessing the images.

3. [RSNA-Exploratory Data Analysis Part 3](https://github.com/takmanman/RSNA-Intercranial-Hemorrhage-Detection/blob/master/RSNA-Exploratory%20Data%20Analysis%20Part%203.ipynb) In this notebook, we will look at the DICOM images with their selected windows. Images from each categories: normal, epidural, intrparenchymal, intraventricular, subarachnoid and subdural will be shown.

## Data Preprocessing

The type of data preprocessing required is entirely dependent of the problem we are tackling and the approach we are going to use. In my approach for this project, I applied the selected window to each image and then save it in as a .npy file. The selected window is bascially a linear transform operation applied on the pixel values. It is determined by a radiologist to make the region of interest (e.g. blood, bone, cavity, etc) most easy to see. It is defined by four values: rescale intercept, rescale slope, window center and window width, which are encoded in the DICOM image file. These values were extracted and organized into a data table in [RSNA-Exploratory Data Analysis Part 2](https://github.com/takmanman/RSNA-Intercranial-Hemorrhage-Detection/blob/master/RSNA-Exploratory%20Data%20Analysis%20Part%202.ipynb)

## Model Building and Training

## Inferencing

## Directory Structure
