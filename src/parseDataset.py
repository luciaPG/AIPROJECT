import csv
from collections import namedtuple

import pandas as pd

# Collects information from the csv using the python collection format namedTuple
# so we can get a tuple for each row and work with them
def data_collection(file):
    with open(file, 'r') as f:
        lines = csv.reader(f, delimiter=',')
        next(f)
        houses = []
        house = namedtuple("house", ["beds","baths","size","size_units","lot_size","lot_size_units","zip_code","price"])

        for beds,baths,size,size_units,lot_size,lot_size_units,zip_code,price in lines:

            houses.append(house(int(beds),parser_float(baths),int(size),size_units,
                                parser_float(lot_size),lot_size_units,int(zip_code),int(price)))
        return houses


def parser_float(fl):
    if(fl == ""):
        return None
    else:
        return float(fl)

#Group of functions to work with the data from seattlehouses.csv using pandas libraries
def machine_learning(df):
    #print("Shape: Rows="+str(df.shape[0])," Columns="+str(df.shape[1]))

    print("Info:", df.info()) #-> General info about the dataset

    #print(df.index) -> Gives u the index of the data frame
    #print(df.columns) -> gives u the headers of the columns
    #print(df.values) -> provides examples of the values in rows

    #print(df.describe()) -> stats like mean, count, min, max of each column
    #print(df.sort_values(by="price")) # -> sort the data according to any column
    #print("Beds column:\n",df["beds"]) # -> gives u only a column of the table, type: pandas.Series

    # Filter data according to boolean statement
    #print("Houses with more than 5 baths:\n","Nº houses: ",df[df.baths > 5].shape[0],"\n",df[df.baths > 5])


