# EE_551_Project
EE 551 Final Project



### Dependences
- SKLearn
- Pandas
- Numpy
- Keras
    - Tensorflow Backend



## Purpose:
- House Prices: Advanced Regression Techniques
- Will participate in the following challenge:

#### Description 

Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

#### Goal

It is your job to predict the sales price for each house. For each Id in the test set, you must predict the value of the SalePrice variable. 

#### Metric

Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)
 
#### Data for this competition comes from the following sources:
        
The Ames Housing dataset was compiled by Dean De Cock for use in data science education. It's an incredible alternative for data scientists looking for a modernized and expanded version of the often cited Boston Housing dataset. 



## Steps taken in Project:
- Unpack Data
- Prepare data for multivariate models 
    - Remove categorical categories and replace with one-hot encoded categories
    - Fill all 'NaN' entries with the average from that column (In an ideal situation, this would not have an effect on the model)
- Create train and test variables
    - Kaggle provides all train variables
    - Kaggle does not provide the test output to keep competitors honest
- Run models
    - 3 different models were used:
        1. Simple Linear Model (SKLearn)
        2. Least Absolute Shrinkage Selector Operator (LASSO) Regression (SKLearn)
        3. Keras Regressor (Wrapper for SKLearn)
        
## Results

### 1. Simple Linear Regression Model with SKLearn
Linear regression is a statistical model that estimates a scalar response based on one or more explanatory variables. 
The simple Linear Model with SKLearn gave me a score of 0.4639 on Kaggle.

### 2. LASSO Regression Model with SKLearn
Lasso regression is a statistical model that performs variable selection and regularization to improve upon a linear model.
The LASSO Regression Model with SKLearn gave me a score of 0.4611 on Kaggle, not a big improvement.

### 3. Keras Regressor (Wrapper for SKLearn)
Deep learning is a machine learning method using a network of neurons to teach a model. Rather than just one estimation made by a statistical model, a deep learning model has several layers of estimation.
The first Keras Regression model (abbreviated KR for this description) is a neural network with two layers. This yielded me a score of 0.207 on Kaggle. The second, and best KR model I created, is a neural network with three layers. The following is a description of the model:


![Alt text](model_plot.png?raw=true "Keras Model Visualization")



