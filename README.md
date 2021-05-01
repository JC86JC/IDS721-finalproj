# IDS721 Final Project-Airbnb housing price prediction model

## Introduction
Airbnb is one of the biggest vacation rental online marketplace company in United States. When renting for a house/apartment, people usually care a lot about the price.  In this project, our group tried to predict the price level of Airbnb houses using random forest model, and then display the test result in a flask app using Google Cloud Platform. 

We got our dataset from this [Kaggle competition](https://www.kaggle.com/c/duke-cs671-fall20-airbnb-pricing-data). This dataset contains the Airbnb housing information and price levels in Buenos Aires, and is used for four-class classification projects. There are 9681 observations in total. We trained and tuned a random forest model on a training set with 6502 rows, and test the model on the remaining 3000 observations. The evaluation metric we used is a confusion matrix, and can displayed in the /report route. In order to verify Elastic Scale-Up Performance of our project, we used Locust load test and scaled up to around 1000 requests per second, which came with 100% success rate.

## Model Diagram

![image](https://user-images.githubusercontent.com/54278431/116767789-6cfa5880-aa00-11eb-8e9a-18c7190f67a5.png)


## Model Detail
- Machine Learning Framework: Scikit-learn
- Model: Random Forest Classifier
- Platform: Flask + Google App Engine
- Elastic Scale-Up Performance Verification: Locust load test with 1000 users.

## Install & Use on GCP
- git clone
- create a virtualenv and source
- run `make install`

### Running app:
#### Use parameters trained and tuned by us:

- to run it locally: run `python main.py`
- to deploy it through public url: run `gcloud app deploy`

#### Use your own dataset to train and test:

1. replace the `train.csv` and `test.csv` with your own data
2. run `python model.py`
3. same as the previous section

### To predict and print the confusion matrix:

- go to the main route and add `/report`
