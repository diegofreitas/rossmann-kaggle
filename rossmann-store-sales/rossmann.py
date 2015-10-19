from matplotlib.backends.qt_editor.formlayout import fedit
from sklearn.dummy import DummyRegressor

__author__ = 'diego.freitas'

import csv

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.metrics import *
from sklearn.svm import SVR
from sklearn.preprocessing import *
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import *
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from utils import rmspe, rmspe_xg, rmspe_score
import scipy
from sklearn.grid_search import GridSearchCV

from sklearn.svm import SVR

rng = np.random.RandomState(1)


sh_encoder = LabelEncoder()
ass_encoder = LabelEncoder()
st_encoder = LabelEncoder()
promo_interval_encoder = LabelEncoder()
type_assort_encoder = LabelEncoder()
p1_encoder = LabelEncoder()
p2_encoder = LabelEncoder()
p3_encoder = LabelEncoder()
p4_encoder = LabelEncoder()
features_scaler = MinMaxScaler()
features_scaler.fitted = False

store_id_hasher = FeatureHasher(n_features=5, input_type='string', non_negative=True)
store_id_hasher.fitted = False

sales = MinMaxScaler()
sales.fitted = False
sale_means = None

#'Store_a', 'Store_b', 'Store_c', 'Store_d', 'Store_e',
features = ['Store_a', 'Store_b', 'Store_c', 'Store_d', 'Store_e', "DayOfWeek", "Open", "Promo", "StateHoliday", 'StoreType_Assortment',
            "SchoolHoliday", "StoreType", "Assortment", "CompetitionDistance", "PromoInterval", "Promo2", 'Promo','CompetitionOpen','PromoOpen','p_1','p_2','p_3','p_4',
            'Sales_Mean_Promo', 'month', 'woy']

categorical_features = [
    header in ["DayOfWeek", "StateHoliday", "Assortment", "StoreType", "PromoInterval", 'month', 'woy','p_1','p_2','p_3','p_4', 'StoreType_Assortment'] for header in
    features]
one_hot_encoder = OneHotEncoder(categorical_features=categorical_features)


def load_data(file="train.csv"):
    global sale_means
    df = pd.read_csv(file, dtype={
        'DayOfWeek': np.int,
        'Sales': np.float64,
        'Store': np.int,
        'SchoolHoliday': np.int
    }, parse_dates=['Date'])
    store = pd.read_csv("store.csv", low_memory=False)
    df = pd.merge(df, store, on='Store')
    # df.loc[(df.Open.isnull() & df.Sales > 0), 'Open'] = 1
    df.fillna(0, inplace=True)

    df['StoreType_Assortment'] = df['StoreType'] + df['Assortment']
    df['year'] = df.Date.apply(lambda x: x.year)
    df['month'] = df.Date.apply(lambda x: x.month)
    df['woy'] = df.Date.apply(lambda x: x.weekofyear)

    df['CompetitionOpen'] = 12 * (df.year - df.CompetitionOpenSinceYear) + (df.month - df.CompetitionOpenSinceMonth)
    df['CompetitionOpen'] = df.CompetitionOpen.apply(lambda x: x if x > 0 else 0)

    df['PromoOpen'] = 12 * (df.year - df.Promo2SinceYear) + (df.woy - df.Promo2SinceWeek) / float(4)
    df['PromoOpen'] = df.PromoOpen.apply(lambda x: x if x > 0 else 0)

    df['p_1'] = df.PromoInterval.apply(lambda x: x[:3] if type(x) == str else 0)
    df['p_2'] = df.PromoInterval.apply(lambda x: x[4:7] if type(x) == str else 0)
    df['p_3'] = df.PromoInterval.apply(lambda x: x[8:11] if type(x) == str else 0)
    df['p_4'] = df.PromoInterval.apply(lambda x: x[12:15] if type(x) == str else 0)

    if not hasattr(sh_encoder, "classes_"):
        df = df[df["Open"] != 0]
        sh_encoder.fit(df["StateHoliday"])
        ass_encoder.fit(df["Assortment"])
        st_encoder.fit(df["StoreType"])
        type_assort_encoder.fit(df['StoreType_Assortment'])
        promo_interval_encoder.fit(df["PromoInterval"])
        p1_encoder.fit(df['p_1'])
        p2_encoder.fit(df['p_2'])
        p3_encoder.fit(df['p_3'])
        p4_encoder.fit(df['p_4'])
        df = df.loc[df.Sales > 0]
        sales.fit(df["Sales"])
        sales.fitted = True
        store_id_hasher.fit_transform(df.Store.astype(np.str))

    df['StoreType_Assortment'] = type_assort_encoder.transform(df['StoreType_Assortment'])
    df["StateHoliday"] = sh_encoder.transform(df["StateHoliday"])
    df["Assortment"] = ass_encoder.transform(df["Assortment"])
    df["StoreType"] = st_encoder.transform(df["StoreType"])
    df["PromoInterval"] = promo_interval_encoder.transform(df["PromoInterval"])
    df["p_1"] = p1_encoder.transform(df["p_1"])
    df["p_2"] = p2_encoder.transform(df["p_2"])
    df["p_3"] = p3_encoder.transform(df["p_3"])
    df["p_4"] = p4_encoder.transform(df["p_4"])

    if 'Sales' in df.columns:
        sale_means = df.groupby(['Store', 'DayOfWeek', 'Promo']).mean().Sales
        sale_means = sale_means.reset_index()
        sale_means.rename(columns={'Sales': 'Sales_Mean_Promo'}, inplace=True)

    df = pd.merge(df, sale_means, on=['Store', 'DayOfWeek', 'Promo'], how='left')
    df.fillna(0, inplace=True)


    # Test data set does not have  Sales
    if not 'Sales' in df.columns:
        df["Sales"] = np.zeros(df.shape[0])
        #when test set is merged the order is messed up
        df.sort(['Id'], inplace=True)

    #sales_comptdistance = scaler.transform(df[["CompetitionDistance", "Sales"]].values)

    if not features_scaler.fitted:
        features_scaler.fit(df[["CompetitionDistance", "Sales_Mean_Promo"]])
        features_scaler.fitted = True

    df[["CompetitionDistance", "Sales_Mean_Promo"]] = features_scaler.transform(
        df[["CompetitionDistance", "Sales_Mean_Promo"]])
    df["Sales"] = sales.transform(df["Sales"])

    store_id = store_id_hasher.transform(df.Store.astype(np.str)).toarray()
    store_id_df = pd.DataFrame(columns=['Store_a', 'Store_b', 'Store_c', 'Store_d', 'Store_e'], data=store_id)
    df = df.join(store_id_df)

    print df.head()
    if not hasattr(one_hot_encoder, "feature_indices_"):
        X = one_hot_encoder.fit(df[features].values)
    X = one_hot_encoder.transform(df[features].values).toarray()
    Y = df["Sales"]
    return X, Y

X, Y = load_data()
#poly = PolynomialFeatures(degree=2)
#X = poly.fit_transform(X)

#Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.1, random_state=42)

print "Trainning"
#regressor = linear_model.Lasso()
#regressor = ExtraTreesRegressor(n_estimators=100, warm_start=True)

#regressor = GridSearchCV(RandomForestRegressor(n_estimators=20, warm_start=True, n_jobs=6), param_grid={'C': [1, 10]}, scoring=rmspe_score, n_jobs=6, cv=)
#regressor = linear_model.RidgeCV(scoring = rmspe_score, cv=4) 0.13
regressor = RandomForestRegressor(n_estimators=100, warm_start=True, verbose=5, n_jobs=6, random_state=rng)
#regressor = KernelRidge() #Memory Error
#regressor = GradientBoostingRegressor(n_estimators=10,warm_start=True, verbose=4)
#regressor = SVR(kernel='rbf', cache_size=1000)
#regressor.fit(Xtrain, Ytrain)

#predicted = regressor.predict(Xtest)

#print "RMSPE", rmspe(sales.inverse_transform(Ytest), sales.inverse_transform(predicted))
#exit()
print "Refit"
regressor.fit(X, Y)

X, Y = load_data("test.csv")

predicted = sales.inverse_transform(regressor.predict(X))


with open('submission_rossman.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["Id", "Sales"])
    for index in xrange(0, len(predicted)):
        writer.writerow([index + 1, predicted[index]])

