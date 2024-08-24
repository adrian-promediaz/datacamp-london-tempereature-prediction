from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import mlflow


def impute_and_scale(data_frame, imputer_strategy, feature_selection, target_column):
    cleaned_data_frame = data_frame.dropna(subset=target_column)
    X = cleaned_data_frame[feature_selection]
    y = cleaned_data_frame[target_column]

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    # Impute misising value
    imputer = SimpleImputer(strategy=imputer_strategy)
    # Fit on the training data
    X_train = imputer.fit_transform(X_train)
    # Transform on the test data
    X_test = imputer.transform(X_test)


    # Scale the data
    scaler = StandardScaler()
    # Fit on the training data
    X_train = scaler.fit_transform(X_train)
    # Transform on the test data
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def linear_regression (X_train, X_test, y_train, y_test):
    lin_reg = LinearRegression().fit(X_train, y_train)
    y_pred_lin_reg = lin_reg.predict(X_test)
    lin_reg_rmse = mean_squared_error(y_test, y_pred_lin_reg, squared = False)
    return lin_reg, lin_reg_rmse

def tree_regression (X_train, X_test, y_train, y_test, depth):
    tree_reg = DecisionTreeRegressor(random_state=42, max_depth=depth).fit(X_train, y_train)
    y_pred_tree_reg = tree_reg.predict(X_test)
    tree_reg_rmse = mean_squared_error(y_test, y_pred_tree_reg, squared = False)
    return tree_reg, tree_reg_rmse

def forest_regression (X_train, X_test, y_train, y_test, depth):
    forest_reg = RandomForestRegressor(random_state=42, max_depth=depth).fit(X_train, y_train)
    y_pred_forest_reg = forest_reg.predict(X_test)
    forest_reg_rmse = mean_squared_error(y_test, y_pred_forest_reg, squared=False)
    return forest_reg, forest_reg_rmse