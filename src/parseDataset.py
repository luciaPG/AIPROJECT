import pandas as pd



def initialData(df):
    #print("Shape: Rows="+str(df.shape[0])," Columns="+str(df.shape[1]))

    #print("Info:", df.info()) #-> General info about the dataset

    #print(df.index) -> Gives u the index of the data frame
    #print(df.columns) -> gives u the headers of the columns
    #print(df.values) -> provides examples of the values in rows

    #print(df.describe()) -> stats like mean, count, min, max of each column
    #print(df.sort_values(by="price")) # -> sort the data according to any column
    print(df["beds"]) # -> gives u only a column of the table, type: pandas.Series



