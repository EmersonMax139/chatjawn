import pandas as pd
import numpy as np 

train_ratio = 0.1 

def prepare_csv():
    train_ds = pd.read_csv("data.csv", error_bad_lines=False)
    # Get rid of newlines
    train_ds = train_ds.str.replace("\n", "")
    
    index = np.arrange(train_ds.shape[0])
    np.random.shuffle(index)

    # Split data  | train_ratio/100 |
    train_ratio = int(len(index) * train_ratio)
    train_ds.iloc[index[train_ratio:], :].to_csv(
        "cache/dataset_train.csv", index=False)
    train_ds.iloc[index[:train_ratio], :].to_csv(
        "cache/dataset_val.csv", index=False)

    test_ds = pd.read_csv("data.csv")
    test_ds = \
        test_ds.str.replace("\n", " ")
    test_ds.to_csv("cache/dataset_test.csv", index=False)

prepare_csv()    