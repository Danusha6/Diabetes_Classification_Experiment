import pandas as pd

def load_diabetes_data(path):
    return pd.read_csv(path)