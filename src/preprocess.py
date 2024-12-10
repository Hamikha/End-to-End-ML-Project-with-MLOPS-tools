import pandas as pd
import sys
import yaml
import os

## Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))['preprocess']

def preprocess(input_path,output_path):
    data = pd.read_csv(input_path)

    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    data.to_csv(output_path,header=None,index=None)

    print(f"Preprocessed Data saved to {output_path}")


if __name__ == "__main__":
    input_path = params['input']
    output_path = params['output']
    preprocess(input_path,output_path)



    

