import pandas as pd

from src.parseDataset import initialData

df = pd.read_csv("C:/Users/User/eclipse-workspace/AIPROJECT/data/seattlehouses.csv")
def initialDataTest(df):
    initialData(df)


initialDataTest(df)


