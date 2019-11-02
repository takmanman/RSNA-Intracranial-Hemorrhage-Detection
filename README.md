# RSNA Intracranial Hemorrhage Detection

The code in this repo demonstrates how to build a baseline deep learning model to detect different types of intracranial hemorrhage from CT scan images. The training data is from a Kaggle competition [RSNA Intracranial Hemorrhage Detection] (https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/). I will go through the usual steps applying data science techniques to problem solving, which are exploratory data analysis, model building and training and inferencing.

## Exploratory Data Analysis

As with any data science projects, a large part of effort is in exploratory data analysis (EDA) and data processing. In this particular training dataset, which you can download from the [competition website](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/), there are over 600000 DICOM images. They are all CT scans of the head, of which about one sixth were identified with some sort of hemorrhage. I have divided the EDA into three jupyter-notebooks:
