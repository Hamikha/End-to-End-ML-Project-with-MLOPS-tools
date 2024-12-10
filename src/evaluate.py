import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow

from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/hamikhan9980/machinelearningpipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "hamikhan9980"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "ae1605bf49828ac787a0304a130b61d618a73b99"

##Loading the params file
params = yaml.safe_load(open('params.yaml'))['train']

def evaluate(data_path,model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    mlflow.set_tracking_uri('https://dagshub.com/hamikhan9980/machinelearningpipeline.mlflow')

    ## loading the file from the desk
    model = pickle.load(open(model_path,'rb'))  

    predictions = model.predict(X)

    accuracy = accuracy_score(y,predictions)

    ## Log metrices to mlflow
    mlflow.log_metric("accuracy",accuracy)
    print(f'Model accuracy = {accuracy}')


if __name__ == "__main__":
    evaluate(params['data'],params['model'])

