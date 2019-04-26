#!/usr/bin/env python
# coding: utf-8

# ### Project Steps
# 1. 'Unpack Data'
# 2. Put data into easy to deal with format (Pandas dataframe?)
# 3. Visualize data

# ### Dependencies
# - Sklearn
# - Pandas
# - Numpy
# - Keras
# - Tensorflow

# ### Global variables
# - train_data (dataframe)
# - test data (dataframe)
# - X (training data from train_data [slice of dataframe])
# - Y (training data from train_data [slice of dataframe])
# - X_test (test data from test_data)
# 
# ***No Y_test is given by Kaggle, Kaggle hides the solution to the test data to keep users honest in the competition

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

train_data = pd.read_csv('House_Data/train.csv')
test_data = pd.read_csv('House_Data/test.csv')


# In[2]:


# Creating list of categorical variables

def find_categorical(data):
    list_categorical = []
    for col in list(data.columns):
        if str(data[col].dtype) != 'int64' and str(data[col].dtype) != 'float64':
            list_categorical.append(col)
    return list_categorical

categorical_columns = find_categorical(train_data)


# In[3]:


# Replace categorical variables with quantitative ones (via one hot encoding)
# not using numeric encoding because I would rather train on true and false since the variables have 
#     nothing to do with each other

def replace_categorical(data,categorical_columns):
    for col in categorical_columns:
        
        # create new dataframe from one hot encoding
        # use pd.get_dummies instead of OneHotEncoder because: 
        # https://medium.com/@guaisang/handling-categorical-features-get-dummies-onehotencoder-and-multicollinearity-f9d473a40417
        one_hot = pd.DataFrame(pd.get_dummies(data[col]))
        
        # rename dataframe columns to add the original column name so we know where the new column came from
        one_hot.columns = [str(one_hot_column) + '_' + col for one_hot_column in one_hot.columns]
        
        # join the one-hot column to the original dataframe
        data = data.join(one_hot)
        
        data = data.drop(col,axis=1)
        # return the new table
    return data


# In[4]:


def prep_data(data):
    # fill the NaN with the mean of the column
    data.fillna(data.mean(),inplace=True)
    # Drop the 'Id' column since we already have an index and do not need it messing up our model
    data = data.drop('Id',axis=1)
    return data


# In[5]:


train_data = replace_categorical(train_data,categorical_columns)
train_data = prep_data(train_data)

test_data = replace_categorical(test_data,categorical_columns)
test_data.fillna(test_data.mean(),inplace=True)


# In[6]:


# check for the columns that are in train and aren't in test
def check_leftover(data1,data2):
    non_overlap = []
    for col in data1.columns:
        if col not in list(data2.columns):
            non_overlap.append(col)
    return non_overlap

columns_in_train_not_in_test = check_leftover(train_data,test_data)


# In[7]:


# create train X and Y and test X
def train_data_X_and_Y(data,columns_missing,goal_column):
    X = data.drop(columns_missing,axis=1)
    Y = data[goal_column]
    return X,Y

X,Y = train_data_X_and_Y(train_data,columns_in_train_not_in_test,'SalePrice')
X_test = test_data.drop('Id',axis=1)


# In[8]:


# Create multilinear regression model using sci-kit learn
model = linear_model.LinearRegression()
model.fit(X,Y)
# print('Intercept: \n', model.intercept_)
# print('Coefficients: \n', model.coef_)


# In[9]:


# Create Kaggle submission csv
def create_submission(data, prediction, pred_num, prediction_column='SalePrice'):
    submission = data['Id']
    submission = pd.DataFrame(submission)
    submission[prediction_column] = prediction
    sub_str = 'submission' + str(pred_num) + '.csv'
    submission.to_csv(sub_str,index=False)


# In[10]:


Y_prediction = model.predict(X_test)
create_submission(test_data,Y_prediction,1)


# ### First Trial Summary -- Simple Linear Regression landed me with a score of ~.4639 on Kaggle... Let's see how we can improve

# In[11]:


# Different type of linear model (Least Absolute Shrinkage Selector Operator), which automatically does feature selection

# alpha balances the amount of emphasis given to minimizing RSS vs minimizing sum of square of coefficients
lasso = linear_model.Lasso(alpha=10)
lasso.fit(X,Y)
y_lasso_pred = lasso.predict(X_test)

create_submission(test_data,y_lasso_pred,2)


# ### Second Trial Summary - Lasso Regression landed me with a ~.4611

# In[12]:


# Deep Learning model with Keras

def deep_learning_model():
    model = Sequential()
    # input dimensions is the number of independent variables (all the columns in train_data except for SalePrice)
    # activation = output of node (neuron) = exponential linear unit (after testing it yielded better results)
    model.add(Dense(135, input_dim=270, kernel_initializer='normal', activation='elu'))
    model.add(Dense(1, kernel_initializer='normal'))
    
    # Compile model (configure for training)
    # optimizer 'adam' was chosen because it (on average) is the speediest
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=deep_learning_model, epochs=100, batch_size=5, verbose=0)

estimator.fit(X,Y)
y_keras_pred = estimator.predict(X_test)

create_submission(test_data,y_keras_pred,3)


# ### Third Trial Summary -- Big improvement! Deep learning received a score on Kaggle of 0.207

# In[13]:


# Create build function for KerasRegressor

def deep_learning_model2():
    model = Sequential()
    # input dimensions is the number of independent variables (all the columns in train_data except for SalePrice)
    # activation = output of node (neuron) = exponential linear unit (after testing it yielded better results)
    model.add(Dense(135, input_dim=270, kernel_initializer='normal', activation='elu'))
    model.add(Dense(135, kernel_initializer='normal'))
    model.add(Dense(1, kernel_initializer='normal'))
    
    # Compile model (configure for training)
    # optimizer 'adam' was chosen because it (on average) is the speediest
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# evaluate model with standardized dataset
estimator2 = KerasRegressor(build_fn=deep_learning_model2, epochs=100, batch_size=5, verbose=0)

estimator2.fit(X,Y)
y_keras_pred2 = estimator2.predict(X_test)

create_submission(test_data,y_keras_pred2,4)


# In[14]:


deep_learning_model2().summary()


# ### Fourth Trial-- added another layer to my neural network, which gave me a score on Kaggle of 0.164
