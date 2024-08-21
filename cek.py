import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


weather = pd.read_csv("london_weather.csv")
# converting the date format
weather["date"] = weather["date"].apply(lambda x:pd.to_datetime(str(x), format = '%Y%m%d'))

# taking the month
weather["month"] = weather["date"].dt.month
weather["year"] = weather["date"].dt.year
# weather.head()

sns.lineplot(data=weather, x="year", y="mean_temp")

sns.lineplot(data=weather, x="month", y="mean_temp")

weather_corr = weather.drop(["max_temp","min_temp"], axis=1).corr()
sns.heatmap(weather_corr)


feature_selection = ["global_radiation","sunshine","month","cloud_cover","precipitation","pressure"]
weather = weather.dropna(subset=['mean_temp'])
X = weather[feature_selection]
y = weather[["mean_temp"]]
# X.head()

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# Impute misising value
imputer = SimpleImputer(strategy="mean")
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

# Predict, evaluate, and log the parameters and metrics of your models
for idx, depth in enumerate ([1, 2]):
    run_name = f"run_{idx}"
    print(run_name)
    with mlflow.start_run(run_name = run_name):

        #Create models
        lin_reg = LinearRegression().fit(X_train, y_train)
        tree_reg = DecisionTreeRegressor(random_state=42, max_depth=depth).fit(X_train, y_train)
        forest_reg = RandomForestRegressor(random_state=42, max_depth=depth).fit(X_train, y_train)
        
        #Log models
        mlflow.sklearn.log_model(lin_reg, "lin_reg")
        mlflow.sklearn.log_model(tree_reg, "tree_reg")
        mlflow.sklearn.log_model(forest_reg, "forest_reg")
    
        #Evaluate performance
        y_pred_lin_reg = lin_reg.predict(X_test)
        lin_reg_rmse = mean_squared_error(y_test, y_pred_lin_reg, squared = False)
        
        y_pred_tree_reg = tree_reg.predict(X_test)
        tree_reg_rmse = mean_squared_error(y_test, y_pred_tree_reg, squared = False)
        
        y_pred_forest_reg = forest_reg.predict(X_test)
        forest_reg_rmse = mean_squared_error(y_test, y_pred_forest_reg, squared=False)
        
        #Log performance
        mlflow.log_param("max_depth", depth)
        mlflow.log_metric("rmse_LR", lin_reg_rmse)
        mlflow.log_metric("rmse_TR", tree_reg_rmse)
        mlflow.log_metric("rmse_RF", forest_reg_rmse)
              
  
        
#Search the runs for the experiment's results
experiment_results = mlflow.search_runs()
print(experiment_results)

## This code is to delete the runs in the experiment

# run_ids = experiment_results["run_id"].to_list()
# for run_id in run_ids:
#     mlflow.delete_run(run_id)
