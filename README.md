# Logistic_Regression_Titanic
Basics about Logistic Regression

This file contains the basic of using a Logistic regression without using regularization. (glmnet package is the one that comes with
the options of regularization).

The Titanic dataset is from [Kaggle](https://www.kaggle.com/c/titanic/data).
The concepts here include accuracy of a prediction model, ROC curve, AUC, LOOCV validation and k-fold cross validation.

The results are shown as follows:
```
[1] "The MSE of glm (randomly pick 450 data for training and the rest for test) is  0.191343963553531"
[1] "The accuracy of the logistic regression is 0.808656036446469"
[1] "The area underneath the ROC curve (AUC) is  0.845001105950011"
[1] "The prediction error using LOOCV is  0.143197291864419"
[1] "The prediction error using K-fold cv is  0.142965624808944"
```

Here is the ROC curve:
![alt tag](https://github.com/GeneralBunny/Titanic/edit/master/ROC.jpeg)
