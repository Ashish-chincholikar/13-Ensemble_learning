# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:53:47 2024

@author: Ashish Chincholikar
Assignment
Ensemble learning
"""

"""
1. Business Problem
    for a problem related to healthCare Sector , Diabetes is a condition that can 
    be predicted for any individual based on certain parameters of that person's
    overall health features , we have to develope a model that predicts that wheather
    a person is having the Diabetes or not . this prediction on the features help us 
    enable various analytics data , that a person having similar parameters have more
    chances of developing the diabetes in his near future and hence with this kind of
    insightful data the healthcare experts can ask the individual to take up the neccessary 
    tests to get the diabetes checked and if found positive , immediate medications can 
    be given to the individual 
    
1.1 what is business objective?
    ~To develope a model which predicts the chances of getting diabetes to a person 
    with the given feature or parameters
    ~To identify the Diabetes condition as early as possible and provide the right 
    medication as early as possible
    
1.2 Are there any constraints
    ~Data Collection 
    ~features contributing to diabetes can also be different than the features we are
    assuming and developing model for

2. Create a Data Dictionary 
name of feature 
description 
type 
relevance

1. Pregnancies , No of time the individual has been pregrant , ~ , Relavent data
2. Glucose , Glucose level of that individual , ~ , Relavent data
3. BloodPressure , the blood pressure levels of that indidual , ~  , Relevant data
4. SkinThickness , how thick the skin of that individula is  ,~ , irrelevant
5. Insuline , does the indvisual take insuline , ~ , releavant
6. BMI , BMI of an individual , ~ , relevant
7. Age , Age of that individual , ~ , relevant
8. DiabetesPredigreeFunction , ~ , ~ , irrelevant
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load the dataframe
df = pd.read_csv("C:/Ensemble_Learning/DataSet/Diabeted_Ensemble.csv")

#print the top records of the Dataframe
df.head

#columns of the dataframe
df.columns

#what are the datatypes of the columns
df.dtypes

#5 number summary of the dataframe
df.describe

# check for null values
df.isnull()


# False
df.isnull().sum()
# 0 no null values

""" 
Number of times pregnant           int64
 Plasma glucose concentration      int64
 Diastolic blood pressure          int64
 Triceps skin fold thickness       int64
 2-Hour serum insulin              int64
 Body mass index                 float64
 Diabetes pedigree function      float64
 Age (years)                       int64
 Class variable                   object
dtype: object

"""
##################################################

# Identify the duplicates
duplicate=df.duplicated()
# Output of this function is single columns
# if there is duplicate records output- True
# if there is no duplicate records output-False
# Series is created
duplicate
# False
sum(duplicate)

df.isnull().sum()
df.dropna()
df.columns

# boxplot
# boxplot on Income column
sns.boxplot(df.Number_of_times_pregnant)
# In _Number_of_times_pregnant column 3 outliers 


sns.boxplot(df._Plasma_glucose_concentration)
# In _Plasma_glucose_concentration column 1 outliers

# boxplot on df column
sns.boxplot(df)
# There is outliers on all columns

# Pair-Plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df);
plt.show()

# histplot - show distributions of datasets
sns.histplot(df['Number_of_times_pregnant'],kde=True)
# right skew and the distributed

sns.histplot(df['_Plasma_glucose_concentration'],kde=True)
# left skew and the distributed

sns.histplot(df,kde=True)


# Pair-Plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df);
plt.show()


"""
#Train using Bagging 
from sklearn.ensemble import BaggingClassifier

bag_model = BaggingClassifier(
    base_estimator = DecisionTreeClassifier() , 
    n_estimators=100 , 
    max_samples = 0.8 , 
    oob_score = True,
    random_state = 0
)

bag_model.fit(X_train , y_train)
bag_model.oob_score_

"""
#bag_model.oob_score_
#Out[35]: 0.7534722222222222 
"""
#Note here we are not using test data,using OOB samples results are tested
bag_model.score(X_test , y_test)
# 0.7760416666666666

#Now let us apply cross validation 
bag_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators= 100 , 
    max_samples= 0.8 , 
    oob_score= True , 
    random_state = 0
)

scores = cross_val_score(bag_model, X , y , cv=5)
scores
scores.mean()
#0.7578728461081402

#we can see some improvement in test score with bagging classifier as comp
 """