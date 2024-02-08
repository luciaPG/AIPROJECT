import pandas as pd

from parseDataset import initialData

df = pd.read_csv("data/test_clean.csv")
def initialDataTest(df):
    initialData(df)


initialDataTest(df)



