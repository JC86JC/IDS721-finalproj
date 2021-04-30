# IDS721-finalproj
Final project for IDS 721

Airbnb is one of the biggest vacation rental online marketplace company in United States. When renting for a house/apartment, people usually care a lot about the price.  In this project, our group tried to predict the price level of Airbnb houses using random forest, and then display the test result in a flask app. 

We got our dataset from this [Kaggle competition](https://www.kaggle.com/c/duke-cs671-fall20-airbnb-pricing-data). This dataset contains the Airbnb housing information and price levels in Buenos Aires. There are 9681 observations in total. We trained a random forest model on a training set with 6502 rows, and test the model on the remaining observations. The evaluation metric we used is a confusion matrix, and it is displayed in the /report route.
