import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler


def regression(data_train, data_test):
    X_train = data_train.drop(columns=['price'])
    y_train = data_train['price']
    X_test = data_test.drop(columns=['price'])
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    alphas = np.logspace(-4, -1, 4)
    l1_ratio = np.arange(0.6, 1, 0.1)
    
    ridge = RidgeCV(alphas=alphas)
    lasso = LassoCV(alphas=alphas)
    elastic_net = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratio)
    
    
    ridge.fit(X_train_scaled, y_train)
    lasso.fit(X_train_scaled, y_train)
    elastic_net.fit(X_train_scaled, y_train)
    
    result = {
        "ridge": {
            "alpha": ridge.alpha_,
            "pred": np.round(ridge.predict(X_test_scaled), 2),
            "coefficients": pd.DataFrame({
                "variable": X_train.columns,
                "coef": ridge.coef_
            }).loc[abs(ridge.coef_) > 0.001]
        },
        "lasso": {
            "alpha": lasso.alpha_,
            "pred": np.round(lasso.predict(X_test_scaled), 2),
            "coefficients": pd.DataFrame({
                "variable": X_train.columns,
                "coef": lasso.coef_
            }).loc[lasso.coef_ != 0]
        },
        "elastic_net": {
            "alpha": elastic_net.alpha_,
            "11_ratio": elastic_net.l1_ratio_,
            "pred": np.round(elastic_net.predict(X_test_scaled), 2),
            "coefficients": pd.DataFrame({
                "variable": X_train.columns,
                "coef": elastic_net.coef_
            }).loc[abs(elastic_net.coef_) > 0.001]
        }
    }
    
    return result


data_train = pd.read_csv(".\\data\\data_3\\data_train.csv")
data_test = pd.read_csv(".\\data\\data_3\\data_test.csv")

result = regression(data_train, data_test)
print(result)