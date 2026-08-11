import pandas as pd
import numpy as np
import mlflow 
from sklearn.preprocessing import MinMaxScaler
import joblib as jl
from pathlib import Path
PROJECT_ROOT = Path("/workspace")
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
df=pd.read_csv("/workspace/data/Dataset.txt",delimiter=";")
#cyclic encoding
def cyclic_encoding(df,col,period):
    df[f"{col}_sin"] = np.sin(2*np.pi*df[col]/period)
    df[f"{col}_cos"] = np.cos(2*np.pi*df[col]/period)
    return df

df['date_time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],dayfirst=True)
cols_to_process=df.select_dtypes(include='object').columns.difference(["Date","Time"])
df[cols_to_process]=df[cols_to_process].apply(
    lambda col: pd.to_numeric(col,errors='coerce')
)
df = df.dropna(subset=cols_to_process)  
df['date_time']=pd.to_datetime(df['date_time']) 
df["day_of_year"]=df["date_time"].dt.dayofyear.astype(float)
is_leap_year=df['date_time'].dt.is_leap_year
is_after=df['date_time'].dt.month>2
df.loc[is_leap_year&is_after,'day_of_year']-=1
df=cyclic_encoding(df,"day_of_year",365)
df['year'] = df['date_time'].apply(lambda x: x.year)
df['month'] = df['date_time'].dt.month
df['quarter'] = df['date_time'].dt.quarter
df=cyclic_encoding(df,"quarter",4)
df=cyclic_encoding(df,"month",12)

df=df.loc[:,['date_time','Global_active_power','Global_intensity' ,'Sub_metering_1','Sub_metering_2','Sub_metering_3','year','quarter_sin','quarter_cos','month_sin','month_cos','day_of_year_sin','day_of_year_cos']]
df.sort_values('date_time', inplace=True, ascending=True)
df = df.reset_index(drop=True)
df["weekday"]=df.apply(lambda row: row["date_time"].weekday(),axis=1)
df["weekday"] = (df["weekday"] <    5).astype(int)
dataset = df.select_dtypes(include='number').values #numpy.ndarray
dataset = dataset.astype('float32')
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
#-----------------------------------------------------------------------------------
series = dataset.astype("float32")
split_1 = int(0.7 * len(series))
split_2=int(0.8 * len(series))
train_series = series[:split_1]
val_series=series[split_1:split_2]
test_series  = series[split_2:]

train_series[:,:5]= np.log1p(train_series[:,:5])
test_series[:,:5]=np.log1p(test_series[:,:5])
val_series[:,:5]=np.log1p(val_series[:,:5])

train_series_X= train_series
train_series_y= train_series[:,0].reshape(-1,1)
test_series_X= test_series
test_series_y= test_series[:,0].reshape(-1,1)
val_series_X= val_series
val_series_y= val_series[:,0].reshape(-1,1)

train_series_X = scaler_X.fit_transform(train_series_X)
train_series_y = scaler_y.fit_transform(train_series_y)
val_series_X = scaler_X.transform(val_series_X)
val_series_y = scaler_y.transform(val_series_y)
test_series_X =scaler_X.transform(test_series_X)
test_series_y =scaler_y.transform(test_series_y)
mlflow.set_tracking_uri(
    f"file://{PROJECT_ROOT / 'mlruns'}"
)
with mlflow.start_run():
    mlflow.set_tag("stage", "preprocessing")

    # save data
    jl.dump(scaler_X,DATA_DIR / "scaler_X.pkl")
    jl.dump(scaler_y,DATA_DIR / "scaler_y.pkl")
    pd.to_pickle(train_series_X,DATA_DIR / "train_X.pkl")
    pd.to_pickle(train_series_y,DATA_DIR / "train_y.pkl")
    pd.to_pickle(test_series_X,DATA_DIR / "test_X.pkl")
    pd.to_pickle(test_series_y,DATA_DIR / "test_y.pkl")
    pd.to_pickle(val_series_X,DATA_DIR / "val_X.pkl")
    pd.to_pickle(val_series_y,DATA_DIR / "val_y.pkl")

    mlflow.log_artifact(DATA_DIR / "scaler_X.pkl","data")
    mlflow.log_artifact(DATA_DIR / "scaler_y.pkl","data")
    mlflow.log_artifact(DATA_DIR / "train_X.pkl", "data")
    mlflow.log_artifact(DATA_DIR / "train_y.pkl", "data")
    mlflow.log_artifact(DATA_DIR / "test_X.pkl", "data")
    mlflow.log_artifact(DATA_DIR / "test_y.pkl", "data")
    mlflow.log_artifact(DATA_DIR / "val_X.pkl", "data")
    mlflow.log_artifact(DATA_DIR / "val_y.pkl", "data")
