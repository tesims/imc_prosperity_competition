import os, glob
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_returns(input_folder: str):
    """
    For each CSV in input_folder, read 'log_return', drop nulls, scale to N(0,1).
    Returns dict { product_name: (scaled_returns_array, scaler) }
    """
    ret = {}
    for path in glob.glob(os.path.join(input_folder, "*.csv")):
        product = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)
        series = df["log_return"].dropna().values.reshape(-1,1)
        scaler = StandardScaler().fit(series)
        scaled = scaler.transform(series)
        ret[product] = (scaled, scaler)
    return ret
