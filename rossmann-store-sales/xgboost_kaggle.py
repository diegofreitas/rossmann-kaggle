import pandas as pd
import numpy as np
from sklearn import cross_validation



# Thanks to Chenglong Chen for providing this in the forum
import xgboost as xgb


def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1. / (y[ind] ** 2)
    return w


def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return rmspe


def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return "rmspe", rmspe


# Gather some features
def build_features(features, data):
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
    # Use some properties directly
    features.extend(['Store', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                     'CompetitionOpenSinceYear', 'Promo', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'CompetitionOpen',
                      'month', 'woy', 'PromoOpen'])

    # add some more with a bit of preprocessing
    features.append('SchoolHoliday')
    data['SchoolHoliday'] = data['SchoolHoliday'].astype(float)
    #
    # features.append('StateHoliday')
    #data.loc[data['StateHoliday'] == 'a', 'StateHoliday'] = '1'
    #data.loc[data['StateHoliday'] == 'b', 'StateHoliday'] = '2'
    #data.loc[data['StateHoliday'] == 'c', 'StateHoliday'] = '3'
    #data['StateHoliday'] = data['StateHoliday'].astype(float)

    features.append('DayOfWeek')
    features.append('year')


    features.append('StoreType')
    data.loc[data['StoreType'] == 'a', 'StoreType'] = '1'
    data.loc[data['StoreType'] == 'b', 'StoreType'] = '2'
    data.loc[data['StoreType'] == 'c', 'StoreType'] = '3'
    data.loc[data['StoreType'] == 'd', 'StoreType'] = '4'
    data['StoreType'] = data['StoreType'].astype(float)

    features.append('Assortment')
    data.loc[data['Assortment'] == 'a', 'Assortment'] = '1'
    data.loc[data['Assortment'] == 'b', 'Assortment'] = '2'
    data.loc[data['Assortment'] == 'c', 'Assortment'] = '3'
    data['Assortment'] = data['Assortment'].astype(float)

    #data['StoreType_Assortment'] = data['StoreType'] + data['Assortment']
    data['year'] = data.Date.apply(lambda x: x.year)
    data['month'] = data.Date.apply(lambda x: x.month)
    data['woy'] = data.Date.apply(lambda x: x.weekofyear)

    data['CompetitionOpen'] = 12 * (data.year - data.CompetitionOpenSinceYear) + (data.month - data.CompetitionOpenSinceMonth)
    data['CompetitionOpen'] = data.CompetitionOpen.apply(lambda x: x if x > 0 else 0)

    data['PromoOpen'] = 12 * (data.year - data.Promo2SinceYear) + (data.woy - data.Promo2SinceWeek) / float(4)
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)

    #data['p_1'] = data.PromoInterval.apply(lambda x: x[:3] if type(x) == str else 0)
    #data['p_2'] = data.PromoInterval.apply(lambda x: x[4:7] if type(x) == str else 0)
    #data['p_3'] = data.PromoInterval.apply(lambda x: x[8:11] if type(x) == str else 0)
    #data['p_4'] = data.PromoInterval.apply(lambda x: x[12:15] if type(x) == str else 0)


print("Load the training, test and store data using pandas")
train = pd.read_csv("train.csv", parse_dates=['Date'])
test = pd.read_csv("test.csv", parse_dates=['Date'])
store = pd.read_csv("store.csv")

print("Assume store open, if not provided")
test.fillna(1, inplace=True)

print("Consider only open stores for training. Closed stores wont count into the score.")
train = train[train["Open"] != 0]

print("Join with store")
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

features = []

print("augment features")
build_features(features, train)
build_features([], test)
print(features)

params = {"objective": "reg:linear",
          "eta": 0.3,
          "max_depth": 8,
          "subsample": 1,
          "colsample_bytree": 0.7,
          "silent": 0
}
num_trees = 1000

print("Train a XGBoost model")
val_size = 100000
# train = train.sort(['Date'])
print(train.tail(1)['Date'])
X_train, X_test = cross_validation.train_test_split(train, test_size=0.01)
#X_train, X_test = train.head(len(train) - val_size), train.tail(val_size)
dtrain = xgb.DMatrix(X_train[features], np.log(X_train["Sales"] + 1))
dvalid = xgb.DMatrix(X_test[features], np.log(X_test["Sales"] + 1))
dtest = xgb.DMatrix(test[features])
watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=50, feval=rmspe_xg, verbose_eval=True)

print("Validating")
train_probs = gbm.predict(xgb.DMatrix(X_test[features]))
indices = train_probs < 0
train_probs[indices] = 0
error = rmspe(np.exp(train_probs) - 1, X_test['Sales'].values)
print('error', error)

print("Make predictions on the test set")
test_probs = gbm.predict(xgb.DMatrix(test[features]))
indices = test_probs < 0
test_probs[indices] = 0
submission = pd.DataFrame({"Id": test["Id"], "Sales": np.exp(test_probs) - 1})
submission.to_csv("xgboost_kscript_submission.csv", index=False)
