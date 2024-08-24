import pandas as pd
import numpy as np
import seaborn as sns
import mlflow
import os
import shutil

from preprocessing.preprocessing import date_formatter, plotting, correlation_heat_map
from predictions.predictions import impute_and_scale, linear_regression, tree_regression, forest_regression


weather = pd.read_csv("london_weather.csv")

date_formatter(weather, "date")


plotting(weather, "month", "mean_temp")
plotting(weather, "year", "mean_temp")

correlation_heat_map(weather.drop(["max_temp","min_temp"], axis=1))

features = ["global_radiation","sunshine","month","cloud_cover","precipitation","pressure"]
X_train, X_test, y_train, y_test = impute_and_scale(data_frame=weather, imputer_strategy="mean", feature_selection=features, target_column="mean_temp")

# cleaning the mlruns folder
mlruns_diectory = "/Users/studytube/OTHER_DOCUMENTS/STUDY/GITHUB_REPOS/datacamp-london-tempereature-prediction/mlruns/0"
for item in os.listdir(mlruns_diectory):
    item_path = os.path.join(mlruns_diectory, item)
    if os.path.isdir(item_path):
        shutil.rmtree(item_path)

# Predict, evaluate, and log the parameters and metrics of your models
for idx, depth in enumerate ([1, 5, 10, 20]):
    run_name = f"run_{idx}"
    print(run_name)
    with mlflow.start_run(run_name = run_name):

        #Create models
        lin_reg, lin_reg_rmse = linear_regression(X_train, X_test, y_train, y_test)
        tree_reg, tree_reg_rmse = tree_regression(X_train, X_test, y_train, y_test, depth=depth)
        forest_reg, forest_reg_rmse = forest_regression(X_train, X_test, y_train, y_test, depth=depth)

        # #Log models
        mlflow.sklearn.log_model(lin_reg, "lin_reg")
        mlflow.sklearn.log_model(tree_reg, "tree_reg")
        mlflow.sklearn.log_model(forest_reg, "forest_reg")

        # #Log performance
        mlflow.log_param("max_depth", depth)
        mlflow.log_metric("rmse_LR", lin_reg_rmse)
        mlflow.log_metric("rmse_TR", tree_reg_rmse)
        mlflow.log_metric("rmse_RF", forest_reg_rmse)

#Search the runs for the experiment's results
experiment_results = mlflow.search_runs()
print(experiment_results[["experiment_id","run_id","status","start_time","end_time","params.max_depth",
                          "metrics.rmse_LR","metrics.rmse_TR","metrics.rmse_RF"]])

