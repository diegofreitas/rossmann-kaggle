__author__ = 'diego.freitas'

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from utils import rmspe


# Load the data frame

df = pd.read_csv("train.csv")
print(df)

X = df[["Store", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday"]]

le = LabelEncoder()

X["StateHoliday"] = le.fit_transform(X["StateHoliday"])

Y = df["Sales"]


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.10)

print(X.head(5))
print(Y.head())

lasso = linear_model.Lasso(alpha = 0.1)
lasso.fit(Xtrain, Ytrain)

predicted = lasso.predict(Xtest)


print lasso.coef_


print rmspe(Ytest, predicted)

