import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def date_formatter(data_frame, column_name):
    # converting the date format to YYYY-mm-dd
    data_frame["date"] = data_frame[column_name].apply(lambda x:pd.to_datetime(str(x), format = '%Y%m%d'))

    # taking the month
    data_frame["month"] = data_frame["date"].dt.month
    data_frame["year"] = data_frame["date"].dt.year
    return data_frame

def plotting(data_frame, x_axis, y_axis):
    sns.lineplot(data=data_frame, x=x_axis, y=y_axis).set(title= f"{y_axis} per {x_axis}")
    plt.savefig(f"./models/{y_axis}_per_{x_axis}.png")

def correlation_heat_map(data_frame):
    sns.heatmap(data_frame.corr(), annot=True, fmt=".1f").set(title="Correlation between variables")
    plt.gcf().set_size_inches(10,10)
    plt.savefig(f"./models/Correlation between variables.png")