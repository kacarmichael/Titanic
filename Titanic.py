# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 23:20:00 2020

@author: Aaron
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

import re

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import glm

import scipy.stats as sts
from scipy.stats.stats import pearsonr

from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix

#from pandasgui import show

os.chdir(r'C:\Users\Aaron\Desktop\Personal Data Projects\Titanic')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

combined = train.append(test).reset_index()

combined.describe()

#Create column containing passenger title
title = re.compile('[A-Z]{1}[a-z]+\.')
combined['title'] = [title.findall(name) for name in combined.Name]
combined['title'] = [x[0] for x in combined['title']]


title_count = combined['title'].value_counts().sort_values(ascending=True)

title_count.plot.barh()
plt.show()

low_titles = ['Mr.', 'Miss.', 'Mrs.', 'Ms.']

combined['vip_status'] = np.where(combined['title'].isin(low_titles), 0, 1)

combined['Sex'] = np.where(combined['Sex'] == 'female', 0, 1)

combined['fam_size'] = combined['Parch']+combined['SibSp']

combined.vip_status.value_counts().sort_values(ascending=True).plot.barh()
plt.show()

combined.fam_size.hist()

# temp=combined.describe()

# age_predictors = ['Pclass', 'Sex', 'SibSp', 'Parch']

# age_reg = smf.ols('Age ~ Pclass + Sex + SibSp + Parch', data = combined).fit()
# age_reg.summary()

# #Checking assumptions
# #Linearity
# for pred in age_predictors:
#     combined.plot.scatter(x=pred, y='Age')

# plt.scatter(age_reg.fittedvalues, age_reg.resid)

age_imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
combined['Age'] = age_imp_mean.fit_transform(combined[['Age']])
combined['Age'] = np.around(combined['Age'])
 

age_young = range(0, 13)
age_teen = range(13, 21)
age_adult = range(21, 41)
age_mid = range(41, 61)
age_elder = range(61, 100)
combined['age_categ'] = None
for idx in range(len(combined)):
    if combined['Age'][idx] in age_young:
        combined['age_categ'][idx] = 'young'
    if combined['Age'][idx] in age_teen:
        combined['age_categ'][idx] = 'teen'
    if combined['Age'][idx] in age_adult:
        combined['age_categ'][idx] = 'adult'
    if combined['Age'][idx] in age_mid:
        combined['age_categ'][idx] = 'mid'
    if combined['Age'][idx] in age_elder:
        combined['age_categ'][idx] = 'elder'
        
new_train = combined.iloc[0:len(train),]
y_train = new_train['Survived']
x_train = new_train[['Pclass', 'Sex', 'Age', 'vip_status', 'fam_size']]
new_test = combined.iloc[len(train):,]
        
#logistic regression
#Predict Survived based on Pclass, Sex, Age(or age_categ), vip_status, fam_size
formula = 'Survived ~ Pclass+Sex+Age+vip_status+fam_size'

log_reg_model = glm(formula, data=new_train, family = sm.families.Binomial()).fit()
log_reg_model.summary()
new_train['odds_survival'] = log_reg_model.predict(x_train)
new_train['predicted_survival'] = [x >= 0.5 for x in new_train['odds_survival']]
new_train['predicted_survival'] = np.where(new_train['predicted_survival'] == False, 0, 1)
confusion_matrix(new_train['Survived'], new_train['predicted_survival'])  
# TODO:
    
#     Break down survival by age, sex, and class
#     Build Decision Tree
#     Build Categorical Regression
#     SKLearn Neural Net?
        
        