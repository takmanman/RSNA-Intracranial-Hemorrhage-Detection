# RSNA Intracranial Hemorrhage Detection

The code in this repo demonstrates how to build a baseline deep learning model to detect different types of intracranial hemorrhage from CT scan images. The training data is from the Kaggle competition [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/). I will go through the usual steps of data science problem solving, which are exploratory data analysis, data preprocessing, model building and training, and inferencing.

In the training dataset, which you can download from the [competition website](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/), there are over 600000 DICOM images. They are all CT scans of the head, of which about one seventh were identified with some sort of hemorrhage. A validation dataset was used to assess the performance of this baseline model and the inferences are given in the following table:

|          | Normal | Epidural | Intraparenchymal | Intraventricular | Subarachnoid | Subdural  | Any      |
| -------- | ------ |----------| ---------------- | ---------------- | ------------ | --------- | -------- |
| Accuracy |0.959   |0.997     |0.988             |0.992             |0.980         |0.970      |0.959     |
| Precision|0.972   |0.489     |0.885             |0.831             |0.766         |0.739      |0.858     |
| Recall   |0.981   |0.311     |0.753             |0.835             |0.584         |0.693      |0.802     |

## Exploratory Data Analysis

As with any data science projects, a large part of the effort is in the exploratory data analysis (EDA) and data preprocessing.  I have divided the EDA into three jupyter-notebooks:

1. [RSNA-Exploratory Data Analysis Part 1](https://github.com/takmanman/RSNA-Intercranial-Hemorrhage-Detection/blob/master/RSNA-Exploratory%20Data%20Analysis%20Part%201.ipynb) In this notebook, we will look at the accompanying .csv file of the training dataset. We will find out what types of hemorrhage images are include and their relative distribution. We will also look for any oddity. In the end we will convert the data into data table that will be used for training the model.

2. [RSNA-Exploratory Data Analysis Part 2](https://github.com/takmanman/RSNA-Intercranial-Hemorrhage-Detection/blob/master/RSNA-Exploratory%20Data%20Analysis%20Part%202.ipynb) In this notebook, we will look at the DICOM images and extract their technical specifications. We will find out if the images have similar technical attributes such as sizes, pixel spacings, samples per pixel, etc. We will put these information in another data table which will be use for preprocessing the images.

3. [RSNA-Exploratory Data Analysis Part 3](https://github.com/takmanman/RSNA-Intercranial-Hemorrhage-Detection/blob/master/RSNA-Exploratory%20Data%20Analysis%20Part%203.ipynb) In this notebook, we will look at the DICOM images with their selected windows. Images from each categories: normal, epidural, intrparenchymal, intraventricular, subarachnoid and subdural will be shown.

## Data Preprocessing

The type of data preprocessing required is entirely dependent of the problem we are tackling and the approach we are going to use. In my approach for this project, I applied the selected window to each image and then save it in as a .npy file. The selected window is bascially a linear transform applied on the pixel values. It is determined by a radiologist to make the region of interest (e.g. blood, bone, cavity, etc) most easy to see. It is defined by four values: rescale intercept, rescale slope, window center and window width, which are encoded in the DICOM image file. These values were extracted and organized into a data table in [RSNA-Exploratory Data Analysis Part 2](https://github.com/takmanman/RSNA-Intercranial-Hemorrhage-Detection/blob/master/RSNA-Exploratory%20Data%20Analysis%20Part%202.ipynb)

The code for creating the .npy files is in [RSNA-Create .npy Images](https://github.com/takmanman/RSNA-Intracranial-Hemorrhage-Detection/blob/master/RSNA-Create%20npy%20Images.ipynb)

Before we move on to model training, we should create a validation dataset out of the training dataset. As some of the images were identified with mulitple types of hemorrhage, therefore, this is a multi-labeled dataset.  It is not trivial to maintain the exact same label distribution (e.g. same frequency for the combination of 'epidural' and 'intraventricular'). Nevertheless, I tried to make the label frequency as similar as I can between the training and validation datasets.

The code for creating the validation dataset is in [RSNA-Create Validation Dataset](https://github.com/takmanman/RSNA-Intracranial-Hemorrhage-Detection/blob/master/RSNA-Create%20Validation%20Dataset.ipynb)

## Model Building and Training

To build a baseline model, I applied transfer learning to a pre-trained model, the weakly-supervised ResNeXt-101 32x8d (https://github.com/facebookresearch/WSL-Images). Essentially, I replaced the final fully-connected layer of the pre-trained model, which has 1000 outputs, with a fully-connected layer that has six outputs, each corresponding to one of the labels: epidural, intraparenchymal, intraventricular, subarachnoid, subdural, any. Then I fine tuned the model with the training dataset. It takes about 5 hours to make one pass of the entire dataset with a Nvidia GeForce RTX 2060. I only trained for 3 epoches.

The code for building and training the model is in [RSNA-Model Training](https://github.com/takmanman/RSNA-Intracranial-Hemorrhage-Detection/blob/master/RSNA-Model%20Training.ipynb)

## Inferencing

After training the model, I tested its performance with the validation dataset. I then calculated the accuracy, precision and recall for the predictions made regarding the presense of a hmorrahge as well as the type of hemorrahge.

The code for inferencing from the model is in [RSNA-Validation Inferences](https://github.com/takmanman/RSNA-Intracranial-Hemorrhage-Detection/blob/master/RSNA-Validation%20Inferences.ipynb)

The inferences from the validation dataset are given in the table at the top of the page:

## Conclusions

If you follow the steps outlined above and run the notebooks accordingly, you should be able to obtain a model of similar performance. There may be some slight differences because the validation dataset is created with a random seed, but overall the resulting model should be similar.

There are many possible improvements that can be made to this baseline model. For one, the dataset is quite imbalanced with the majority of the images assessed as normal, and the resulting model would bias towards the majority class. This could be alleviated by weighing the training images according to their ratio in the dataset.

Finally, the file structure of this project is very straight forward. I simply put all the notebooks, table (stored as .pkl files) and model (stored as .pth files) in the same level.

```bash
├──stage-1-train-images\ (images downloaded from competition website)
├──stage-1-train-images-npy\ (images created by RSNA-Create npy Images.ipynb)
├──stage-1-train.csv (downloaded from competition website)
├──RSNA-Exploratory Data Analysis Part 1.ipynb
├──RSNA-Exploratory Data Analysis Part 2.ipynb
├──RSNA-Exploratory Data Analysis Part 3.ipynb 	
├──RSNA-Create npy Images.ipynb
├──RSNA-Create Validation Dataset.ipynb
├──RSNA-Model Training.ipynb
├──rsna-data-table.pkl (created by RSNA-Exploratory Data Analysis Part 1.ipynb)
├──rsna-dicom-table.pkl (created by RSNA-Exploratory Data Analysis Part 2.ipynb)
├──rsna-train-table.pkl (created by RSNA-Create Validation Dataset.ipynb)
└──rsna-valid-table.pkl (created by RSNA-Create Validation Dataset.ipynb)
```
