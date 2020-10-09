# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:51:16 2020

@author: sunilp
"""

import pandas as pd
import pickle

datadf  = pd.read_csv('hiring.csv')
datadf['experience'].fillna(0, inplace = True )
datadf['test_score'].fillna(datadf['test_score'].mean(), inplace = True )


#in the above data salary is the dependent variable(Y)
#other columns are the independent variables(X)
 
#seperating the x values independed values from data
X = datadf.iloc[:,:3]


#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))


#select the dependendt values from the data
y = datadf.iloc[:,-1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
res = regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))




