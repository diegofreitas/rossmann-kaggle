from matplotlib.backends.qt_editor.formlayout import fedit
from sklearn.dummy import DummyRegressor

__author__ = 'diego.freitas'

import csv

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.metrics import *
from sklearn.svm import SVR
from sklearn.preprocessing import *
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import *
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from utils import rmspe, rmspe_xg

rng = np.random.RandomState(1)

sh_encoder = LabelEncoder()
ass_encoder = LabelEncoder()
st_encoder = LabelEncoder()
promo_interval_encoder = LabelEncoder()
features_scaler = MinMaxScaler()
features_scaler.fitted = False

sales = MinMaxScaler()
sales.fitted = False
sale_means = None

features = [ "DayOfWeek", "Open", "Promo", "StateHoliday",
                "SchoolHoliday", "StoreType", "Assortment", "CompetitionDistance", "PromoInterval", "Store", "Promo2", 'Promo','Sales_Mean_Promo']

categorical_features = [ header in ["DayOfWeek", "StateHoliday", "Assortment", "StoreType", "PromoInterval"] for header in features]
print categorical_features
one_hot_encoder = OneHotEncoder(categorical_features=categorical_features)

def load_data(file="train.csv"):
    global  sale_means
    df = pd.read_csv(file, dtype={
        'DayOfWeek': np.int,
        'Sales': np.float64,
        'Store': np.int,
        'SchoolHoliday': np.int
    })
    store = pd.read_csv("store.csv", low_memory=False)
    df = pd.merge(df, store, on='Store')
    #df.loc[(df.Open.isnull() & df.Sales > 0), 'Open'] = 1
    df.fillna(0, inplace=True)

    if not hasattr(sh_encoder, "classes_"):
        df = df[df["Open"] != 0]
        sh_encoder.fit(df["StateHoliday"])
    if not hasattr(ass_encoder, "classes_"):
        ass_encoder.fit(df["Assortment"])
    if not hasattr(st_encoder, "classes_"):
        st_encoder.fit(df["StoreType"])
    if not hasattr(promo_interval_encoder, "classes_"):
        promo_interval_encoder.fit(df["PromoInterval"])
    if not sales.fitted:
        df = df.loc[df.Sales > 0]
        sales.fit(df["Sales"])
        sales.fitted = True


    df["StateHoliday"] = sh_encoder.transform(df["StateHoliday"])
    df["Assortment"] = ass_encoder.transform(df["Assortment"])
    df["StoreType"] = st_encoder.transform(df["StoreType"])
    df["PromoInterval"] = promo_interval_encoder.transform(df["PromoInterval"])

    if 'Sales' in df.columns:
        sale_means = df.groupby(['Store', 'DayOfWeek', 'Promo']).mean().Sales
        sale_means = sale_means.reset_index()
        sale_means.rename(columns={'Sales': 'Sales_Mean_Promo'}, inplace=True)

    df = pd.merge(df, sale_means, on = ['Store','DayOfWeek','Promo'], how='left')
    df.fillna(0, inplace=True)


    #Test data set does not have  Sales
    if not 'Sales' in df.columns:
        df["Sales"] = np.zeros(df.shape[0])
        #when test set is merged the order is messed up
        df.sort(['Id'], inplace=True)

    #sales_comptdistance = scaler.transform(df[["CompetitionDistance", "Sales"]].values)

    if not features_scaler.fitted:
        features_scaler.fit(df[["CompetitionDistance","Sales_Mean_Promo"]])
        features_scaler.fitted = True
    df[["CompetitionDistance","Sales_Mean_Promo"]] = features_scaler.transform(df[["CompetitionDistance","Sales_Mean_Promo"]])
    df["Sales"] = sales.transform(df["Sales"])

    print df.head()

    if not hasattr(one_hot_encoder,"feature_indices_"):
        X = one_hot_encoder.fit(df[features].values)
    X = one_hot_encoder.transform(df[features].values).toarray()
    Y = df["Sales"]

    return X, Y


X, Y = load_data()
#poly = PolynomialFeatures(degree=2)
#X = poly.fit_transform(X.values)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=42)

print "Trainning"
#regressor = linear_model.Lasso()
regressor = ExtraTreesRegressor(n_estimators=100)
#regressor = linear_model.Ridge()

regressor.fit(Xtrain, Ytrain)

predicted = regressor.predict(Xtest)

print "RMSPE", rmspe(sales.inverse_transform(Ytest), sales.inverse_transform(predicted))

#print "Refit"
#regressor.fit(X, Y)

X, Y = load_data("test.csv")

predicted = sales.inverse_transform(regressor.predict(X))


with open('submission_rossman.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["Id", "Sales"])
    for index in xrange(0, len(predicted)):
        writer.writerow([index + 1, predicted[index]])

