# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:36:49 2024

@author: 91721
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load the dataframe
df = pd.read_csv("C:\Ensemble_Learning\DataSet\Tumor_Ensemble.csv")

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

df.dtypes
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

