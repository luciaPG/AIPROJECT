import pandas as pd

from src.parseDataset import *

file = "C:/Users/User/eclipse-workspace/AIPROJECT/data/seattlehouses.csv"
df = pd.read_csv(file)
tuple_list = data_collection(file)

def machine_learning_test(df):
    machine_learning(df)

def data_collection_test():
    for house in tuple_list[:3]:
        print(house)

#machine_learning_test(df) # Group of functions to work with our data using pandas library
data_collection_test() #Gets a list of namedTuples from seattlehouse.csv


