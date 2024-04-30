# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.

2. Read the dataset.

3. Define X and Y array.

4. Define a function for sigmoid, loss, gradient and predict and perform operations.

## Program:
```
Developed by: SUDHIR KUMAR .R
RegisterNumber:  212223230221
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")

dataset

dataset=dataset.drop("sl_no",axis=1)
dataset=dataset.drop("salary",axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')

dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset

X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values

Y

theta=np.random.randn(X.shape[1])

y=Y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta
    
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
    
y_pred = predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)

print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)

print(y_prednew)
```
## Output:
### Read the file and display

![324679994-1f016359-1822-4e61-a4bd-2e42b29b2193](https://github.com/Sudhirr5/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139332214/bdcfdb46-e535-4316-bfdb-a985c8caacc5)

### Categorizing columns

![324680196-803932f9-6766-466c-b982-4a5d4a7f8116](https://github.com/Sudhirr5/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139332214/f3d5c89d-0739-41cf-8cab-a58e803dadd3)

### Labelling columns and displaying dataset

![324680578-3652a512-7da7-424c-9669-89e9451fbdcd](https://github.com/Sudhirr5/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139332214/c270b47a-fc01-4b3a-ab99-a149e9fc8731)

### Display dependent variable

![324680688-e7b177aa-1745-4be5-8193-8ffd8f529775](https://github.com/Sudhirr5/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139332214/eb7bb99b-eec3-4e45-a555-35cd22cb12cd)

### Printing accuracy

![324680804-98088e79-96c4-4b0c-a6fe-9375a8f80849](https://github.com/Sudhirr5/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139332214/120b9b64-97c0-42e0-afc6-622483a25286)

### Printing Y

![324680946-4ad464f1-fd31-45c4-8b6e-200e80a56472](https://github.com/Sudhirr5/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139332214/738f75a8-8afe-4153-9abe-bf176395368f)

### Printing y_prednew

![324681010-66ed7bc1-6b4b-4fea-a1e6-7bca7dad684b](https://github.com/Sudhirr5/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139332214/765d0422-a22a-4bcd-876b-fe2413325cf3)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

