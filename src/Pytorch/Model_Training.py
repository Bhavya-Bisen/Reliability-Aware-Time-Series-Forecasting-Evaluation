import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import yaml
import torch
import mlflow
import pandas as pd
import numpy 
from tqdm import tqdm
from pathlib import Path
import psutil
import os
PROJECT_ROOT = Path("/workspace")
mlflow.set_tracking_uri(
    f"file://{PROJECT_ROOT / 'mlruns'}"
)
# ---- Config ----
config = yaml.safe_load(open("/workspace/config/config.yaml"))

# get experiment
experiment = mlflow.get_experiment_by_name("Default")

# search runs
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.stage = 'preprocessing'",
    order_by=["start_time DESC"],
    max_results=1
)
run_id = runs.iloc[0].run_id

# Get artifact paths
train_path_X = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="data/train_X.pkl"
)
train_path_y = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="data/train_y.pkl"
)

test_path_X = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="data/test_X.pkl"
)
test_path_y = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="data/test_y.pkl"
)

val_path_X = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="data/val_X.pkl"
)
val_path_y = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="data/val_y.pkl"
)

# Load data
train_series_X = pd.read_pickle(train_path_X)
train_series_y = pd.read_pickle(train_path_y)
test_series_X = pd.read_pickle(test_path_X)
test_series_y = pd.read_pickle(test_path_y)
val_series_X = pd.read_pickle(val_path_X)
val_series_y = pd.read_pickle(val_path_y)



class ForecastDataset(Dataset):
    def __init__(self,series_X,series_y,look_back,horizon):
        self.series_X=series_X
        self.series_y=series_y
        self.look_back=look_back
        self.horizon=horizon
    def __len__(self):
        return len(self.series_X)-self.look_back-self.horizon+1
    def __getitem__(self,idx):
        X=self.series_X[idx:idx+self.look_back,:]
        y=self.series_y[idx+self.look_back:idx+self.look_back+self.horizon,:].squeeze(-1)
        return X,y
train_dataset=ForecastDataset(train_series_X,train_series_y,look_back=config["data"]["look_back"],horizon=config["data"]["horizon"])
test_dataset=ForecastDataset(test_series_X,test_series_y,look_back=config["data"]["look_back"],horizon=config["data"]["horizon"])
val_dataset=ForecastDataset(val_series_X,val_series_y,look_back=config["data"]["look_back"],horizon=config["data"]["horizon"])
train_dataloader=DataLoader(train_dataset,batch_size=config["training"]["batch_size"],shuffle=False)
test_dataloader=DataLoader(test_dataset,batch_size=config["training"]["batch_size"],shuffle=False)
val_dataloader=DataLoader(val_dataset,batch_size=config["training"]["batch_size"],shuffle=False)
device=torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(device)
class ForecastModel(nn.Module):
    def __init__(self,input_size,hidden_size,horizon):
        super().__init__()
        self.horizon=horizon
        self.lstm= nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc=nn.Linear(
            hidden_size,
            self.horizon
        )
    def forward(self,x):
        output, (h,c)=self.lstm(x)
        last_hidden=output[:,-1,:]
        pred=self.fc(last_hidden)
        pred=pred.reshape(-1,self.horizon)
        return pred
input_size = train_series_X.shape[1]
model=ForecastModel(input_size=input_size,hidden_size=config["model"]["lstm_units"],horizon=config["data"]["horizon"]).to(device)
print(model)
batch_size=config["training"]["batch_size"]
optimizer=torch.optim.Adam(model.parameters())
loss=nn.HuberLoss()

def train_loop(dataloader,model,loss_fn,optimizer):
    size=len(dataloader.dataset)
    model.train()
    loss_sum=0
    total_samples=0

    pbar = tqdm(dataloader, desc="Training", leave=False)

    for X, y in pbar:
        X = X.float().to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_batch_size = X.size(0)

        loss_sum += loss.item() * current_batch_size
        total_samples += current_batch_size

        avg_loss = loss_sum / total_samples

        pbar.set_postfix(
            loss=f"{avg_loss:.5f}"
        )

    return avg_loss
            
def test_loop(dataloader,model,loss_fn):
    model.eval()
    test_loss=0
    total_samples=0
    targets=[]
    preds=[]
    pbar = tqdm(dataloader, desc="Training", leave=False)

    with torch.no_grad():
        for X, y in pbar:
            X = X.float().to(device)
            y = y.float().to(device)
            pred = model(X)
            
            current_batch_size = X.size(0)
            test_loss += loss_fn(pred, y).item() *current_batch_size
            total_samples += current_batch_size
            
            preds.extend(pred.cpu().numpy().flatten())
            targets.extend(y.cpu().numpy().flatten())

            avg_loss = test_loss / total_samples

            pbar.set_postfix(loss=f"{avg_loss:.5f}")

    df=pd.DataFrame({"targets":targets,"predictions":preds})
                           
    test_loss /= total_samples
    return test_loss , df

with mlflow.start_run(run_name=config["run_name"]):
    mlflow.log_params(config)
    mlflow.set_tag("stage","training")
    initial_df=pd.DataFrame({})
    middle_df=pd.DataFrame({})
    final_df=pd.DataFrame({})
    for t in range(config["training"]["epochs"]):
        process = psutil.Process(os.getpid())
        print(f"Epoch {t+1} Ram:{process.memory_info().rss/1024**3:.2f}GB\n-------------------------------")
        train_loss=train_loop(train_dataloader, model, loss, optimizer)
        test_loss,df=test_loop(test_dataloader, model, loss)
        mlflow.log_metric("train_loss", train_loss, step=t)
        mlflow.log_metric("test_loss", test_loss, step=t)
        if t==0:
            initial_df=df
        if t==24:
            middle_df=df
        if t==49:
            final_df=df        
    initial_df.to_csv("predictions_initial.csv", index=False)
    middle_df.to_csv("predictions_middle.csv", index=False)
    final_df.to_csv("predictions_final.csv", index=False)
    mlflow.log_artifact("predictions_initial.csv")
    mlflow.log_artifact("predictions_middle.csv")
    mlflow.log_artifact("predictions_final.csv")
    mlflow.pytorch.log_model(model, "model")