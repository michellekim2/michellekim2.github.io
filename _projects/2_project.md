[Untitled1.md](https://github.com/michellekim2/michellekim2.github.io/files/12516595/Untitled1.md)---
layout: page
title: CIS 053 Intro to Machine Learning Final Project
description: Use Python, Scikit to perform feature selection, lasso and ridge regression, cross validation on housing price data
img: assets/img/Final Project CIS 053 (Michelle Kim).jpg
importance: 1
category: work
---

Final project to culminate all content mastered over the summer term. 

Programming Tasks: descriptive statistic and generate plots including correlation heatmap, perform manual analysis of plots for potential relevant features, perform feature selection with Recursive Feature Elimination, build regularized regression model (Lasso and Ridge methods), use K-fold method for cross validation

Produced Final Report concisely presenting data and process.

[Uploading Untitled1.md```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 14:04:37 2023

@author: michellekim
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from pandas import set_option
from pandas import read_csv
from sklearn.linear_model import LinearRegression # import Linear Regression
from sklearn.feature_selection import RFE 
from sklearn.preprocessing import StandardScaler
from numpy import set_printoptions
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

filename = 'realEstate.csv'
data = read_csv(filename)
data1 = data.drop(data.columns[0], axis=1) # drop "No" column
data2 = data1.drop(['X1 transaction date'], axis=1) # drop categorical data


names = ['X2 house age','X3 distance to the nearest MRT station', 'X4 number of convenience stores',	'X5 latitude',	'X6 longitude',	'Y house price of unit area']

# separate features price per square unit area is the target (y values/output)
# and the rest of the columns should be treated as inputs
array = data2.values
X_original = array[:,0:5] # features
Y_original = array[:,5]  # target

# DESCRIPTIVE STATISTICS
pd.options.display.max_columns = None
set_option('display.width', 100)
set_option('display.precision', 1)
description = data2.describe()
print("Descriptive Statistics of the Data: ")
print(description)

plt.figure() # new plot
data2.hist()
plt.title("Histogram")
plt.show()

plt.figure() # new plot
corMat = data2.corr(method='pearson')
print(corMat)
## plot correlation matrix as a heat map
sns.heatmap(corMat, square=True)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.title("CORELATION MATTRIX USING HEAT MAP")
plt.show()

plt.figure()
scatter_matrix(data2)
plt.title("Scatterplot")
plt.show()



```

    Descriptive Statistics of the Data: 
           X2 house age  X3 distance to the nearest MRT station  X4 number of convenience stores  \
    count         414.0                                   414.0                            414.0   
    mean           17.7                                  1083.9                              4.1   
    std            11.4                                  1262.1                              2.9   
    min             0.0                                    23.4                              0.0   
    25%             9.0                                   289.3                              1.0   
    50%            16.1                                   492.2                              4.0   
    75%            28.1                                  1454.3                              6.0   
    max            43.8                                  6488.0                             10.0   
    
           X5 latitude  X6 longitude  Y house price of unit area  
    count      4.1e+02       4.1e+02                       414.0  
    mean       2.5e+01       1.2e+02                        38.0  
    std        1.2e-02       1.5e-02                        13.6  
    min        2.5e+01       1.2e+02                         7.6  
    25%        2.5e+01       1.2e+02                        27.7  
    50%        2.5e+01       1.2e+02                        38.5  
    75%        2.5e+01       1.2e+02                        46.6  
    max        2.5e+01       1.2e+02                       117.5  



    <Figure size 640x480 with 0 Axes>



    
![png](output_0_2.png)
    


                                            X2 house age  X3 distance to the nearest MRT station  \
    X2 house age                                 1.0e+00                                 2.6e-02   
    X3 distance to the nearest MRT station       2.6e-02                                 1.0e+00   
    X4 number of convenience stores              5.0e-02                                -6.0e-01   
    X5 latitude                                  5.4e-02                                -5.9e-01   
    X6 longitude                                -4.9e-02                                -8.1e-01   
    Y house price of unit area                  -2.1e-01                                -6.7e-01   
    
                                            X4 number of convenience stores  X5 latitude  \
    X2 house age                                                    5.0e-02      5.4e-02   
    X3 distance to the nearest MRT station                         -6.0e-01     -5.9e-01   
    X4 number of convenience stores                                 1.0e+00      4.4e-01   
    X5 latitude                                                     4.4e-01      1.0e+00   
    X6 longitude                                                    4.5e-01      4.1e-01   
    Y house price of unit area                                      5.7e-01      5.5e-01   
    
                                            X6 longitude  Y house price of unit area  
    X2 house age                                -4.9e-02                        -0.2  
    X3 distance to the nearest MRT station      -8.1e-01                        -0.7  
    X4 number of convenience stores              4.5e-01                         0.6  
    X5 latitude                                  4.1e-01                         0.5  
    X6 longitude                                 1.0e+00                         0.5  
    Y house price of unit area                   5.2e-01                         1.0  



    
![png](output_0_4.png)
    



    <Figure size 640x480 with 0 Axes>



    
![png](output_0_6.png)
    



```python
# normalize: data distribution is unknown or the data doesn't have Gaussian Distribution# standardize: when the data is being used for multivariate analysis i.e. when we want all the variables of comparable units. It is usually applied when the data has a bell curve

# STANDARDIZATION
scaler1 = StandardScaler().fit(data2) # create scaler object and fit data2
rescaled_data = scaler1.transform(data2) # standardize

# a new data frame with the standardized data
dataStandDf = pd.DataFrame(rescaled_data, columns = names) # new dataframe to hold standardized data


plt.figure() # new plot
dataStandDf.hist()
plt.show()

array = dataStandDf.values
X = array[:,0:5] # features
Y = array[:,5]  # target




```


    <Figure size 640x480 with 0 Axes>



    
![png](output_1_1.png)
    



```python
# RFE

for i in range(1,6):
    NUM_FEATURES = i
    model = LinearRegression()
    rfe = RFE(estimator = model, n_features_to_select = NUM_FEATURES)
    fit = rfe.fit(X, Y) 

    print("Num Features:", fit.n_features_)
    print("Selected Features:", fit.support_)
    print("Feature Ranking:", fit.ranking_)
    # calculate the score for the selected features
    score = rfe.score(X,Y)
    print("Model Score with selected features is: ", score)
    



```

    Num Features: 1
    Selected Features: [False  True False False False]
    Feature Ranking: [3 1 2 4 5]
    Model Score with selected features is:  0.45375427891826703
    Num Features: 2
    Selected Features: [False  True  True False False]
    Feature Ranking: [2 1 1 3 4]
    Model Score with selected features is:  0.49656835105076835
    Num Features: 3
    Selected Features: [ True  True  True False False]
    Feature Ranking: [1 1 1 2 3]
    Model Score with selected features is:  0.5410632980005723
    Num Features: 4
    Selected Features: [ True  True  True  True False]
    Feature Ranking: [1 1 1 1 2]
    Model Score with selected features is:  0.5711351969713696
    Num Features: 5
    Selected Features: [ True  True  True  True  True]
    Feature Ranking: [1 1 1 1 1]
    Model Score with selected features is:  0.5711617064827441



```python
# Build a regularized version of the regression model (use both Lasso and Ridge methods) 
X_train,X_test,y_train,y_test = train_test_split(X, Y, test_size=0.3,random_state=3)
lr = LinearRegression()
lr.fit(X_train, y_train)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>




```python


rr = Ridge(alpha=0.01) # higher the alpha value, alpha = lambda, more restriction on the coefficients; low alpha > closer to the unconstraint case (original MLR)

# linear and ridge regression 
rr.fit(X_train, y_train)
rr100 = Ridge(alpha=100) #  comparison with alpha/lambda value
rr100.fit(X_train, y_train)
train_score=lr.score(X_train, y_train)
test_score=lr.score(X_test, y_test)
Ridge_train_score = rr.score(X_train,y_train)
Ridge_test_score = rr.score(X_test, y_test)
Ridge_train_score100 = rr100.score(X_train,y_train)
Ridge_test_score100 = rr100.score(X_test, y_test)
print("linear regression train score:", train_score)
print("linear regression test score:", test_score)
print( "ridge regression train score low alpha:", Ridge_train_score)
print("ridge regression test score low alpha:", Ridge_test_score)
print("ridge regression train score high alpha:", Ridge_train_score100)
print("ridge regression test score high alpha:", Ridge_test_score100)
plt.plot(rr.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha = 0.01$',zorder=7) # zorder for ordering the markers
plt.plot(rr100.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge; $\alpha = 100$') # alpha here is for transparency
plt.plot(lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.show()

lasso = Lasso()
lasso.fit(X_train,y_train)
train_score=lasso.score(X_train,y_train)
test_score=lasso.score(X_test,y_test)
coeff_used = np.sum(lasso.coef_!=0)
print("training score:", train_score) 
print("test score: ", test_score)
print("number of features used: ", coeff_used)

lasso05 = Lasso(alpha=10, max_iter=1000000)
lasso05.fit(X_train,y_train)
train_score05=lasso05.score(X_train,y_train)
test_score05=lasso05.score(X_test,y_test)
coeff_used05 = np.sum(lasso05.coef_!=0)
print( "training score for alpha=0.5:", train_score05 )
print( "test score for alpha =0.5: ", test_score05)
print( "number of features used: for alpha =0.5:", coeff_used05)

lasso001 = Lasso(alpha=0.01, max_iter=1000000)
lasso001.fit(X_train,y_train)
train_score001=lasso001.score(X_train,y_train)
test_score001=lasso001.score(X_test,y_test)
coeff_used001 = np.sum(lasso001.coef_!=0)
print( "training score for alpha=0.01:", train_score001 )
print( "test score for alpha =0.01: ", test_score001)
print( "number of features used: for alpha =0.01:", coeff_used001)

lasso00001 = Lasso(alpha=0.0001, max_iter=1000000)
lasso00001.fit(X_train,y_train)
train_score00001=lasso00001.score(X_train,y_train)
test_score00001=lasso00001.score(X_test,y_test)
coeff_used00001 = np.sum(lasso00001.coef_!=0)
print( "training score for alpha=0.0001:", train_score00001 )
print( "test score for alpha =0.0001: ", test_score00001)
print( "number of features used: for alpha =0.0001:", coeff_used00001)

lr = LinearRegression()
lr.fit(X_train,y_train)
lr_train_score=lr.score(X_train,y_train)
lr_test_score=lr.score(X_test,y_test)
print( "LR training score:", lr_train_score )
print( "LR test score: ", lr_test_score)
plt.subplot(1,2,1)
plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7) # alpha here is for transparency
plt.plot(lasso05.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.1$') # alpha here is for transparency

plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.subplot(1,2,2)
plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7) # alpha here is for transparency
plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.01$') # alpha here is for transparency
plt.plot(lasso00001.coef_,alpha=0.8,linestyle='none',marker='v',markersize=6,color='black',label=r'Lasso; $\alpha = 0.00001$') # alpha here is for transparency
plt.plot(lr.coef_,alpha=0.7,linestyle='none',marker='o',markersize=5,color='green',label='Linear Regression',zorder=2)
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.tight_layout()
plt.show()

```

    linear regression train score: 0.5563145979243255
    linear regression test score: 0.6038048023415713
    ridge regression train score low alpha: 0.5563145970530934
    ridge regression test score low alpha: 0.6038089813453422
    ridge regression train score high alpha: 0.5379363261069321
    ridge regression test score high alpha: 0.5984294805725647



    
![png](output_4_1.png)
    


    training score: 0.0
    test score:  -0.000634330627290014
    number of features used:  0
    training score for alpha=0.5: 0.0
    test score for alpha =0.5:  -0.000634330627290014
    number of features used: for alpha =0.5: 0
    training score for alpha=0.01: 0.5554509831034635
    test score for alpha =0.01:  0.6066758613370471
    number of features used: for alpha =0.01: 4
    training score for alpha=0.0001: 0.5563145031313953
    test score for alpha =0.0001:  0.6038441485928245
    number of features used: for alpha =0.0001: 5
    LR training score: 0.5563145979243255
    LR test score:  0.6038048023415713



    
![png](output_4_3.png)
    



```python

# set up models for cross validation
NUM_FEATURES = 1
model = LinearRegression()
rfe_1 = RFE(estimator = model, n_features_to_select = NUM_FEATURES)


NUM_FEATURES = 2
rfe_2 = RFE(estimator = model, n_features_to_select = NUM_FEATURES)
fit_2 = rfe_2.fit(X, Y) 
    
NUM_FEATURES = 3
rfe_3 = RFE(estimator = model, n_features_to_select = NUM_FEATURES)

NUM_FEATURES = 4
rfe_4 = RFE(estimator = model, n_features_to_select = NUM_FEATURES)

NUM_FEATURES = 5
rfe_5 = RFE(estimator = model, n_features_to_select = NUM_FEATURES)

```


```python

# prepare models
models = []

models.append(('rfe1', rfe_1))#, RandomForestClassifier(max_depth=2, random_state=0)))
models.append(('rfe2', rfe_2))#, RandomForestClassifier(max_depth=3, random_state=0)))
models.append(('rfe3', rfe_3))
models.append(('rfe4', rfe_4))
models.append(('rfe5', rfe_5))

results = []
names = []
for name, mod in models:
    kfold = KFold(n_splits=10, random_state=7, shuffle = True)
    cv_results = cross_val_score(mod, X, Y, cv=kfold, scoring = 'r2')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
```

    rfe1: 0.457374 (0.111615)
    rfe2: 0.500645 (0.141776)
    rfe3: 0.526938 (0.124293)
    rfe4: 0.577923 (0.119377)
    rfe5: 0.575891 (0.119586)



    
![png](output_6_1.png)
    



```python

```


```python

```
…]()


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Final Project CIS 053 (Michelle Kim).jpg"
" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    
</div>
<div class="caption">
    Above is my written report showing each step of the final project
</div>
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/5.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    This image can also have a caption. It's like magic.
</div>

You can also put regular text between your rows of images.
Say you wanted to write a little bit about your project before you posted the rest of the images.
You describe how you toiled, sweated, *bled* for your project, and then... you reveal its glory in the next row of images.


<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.html path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    You can also have artistically styled 2/3 + 1/3 images, like these.
</div>


The code is simple.
Just wrap your images with `<div class="col-sm">` and place them inside `<div class="row">` (read more about the <a href="https://getbootstrap.com/docs/4.4/layout/grid/">Bootstrap Grid</a> system).
To make images responsive, add `img-fluid` class to each; for rounded corners and shadows use `rounded` and `z-depth-1` classes.
Here's the code for the last row of images above:

{% raw %}
```html
<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.html path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
```
{% endraw %}
