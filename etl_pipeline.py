# import libraries
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

# load messages dataset
messages = pd.read_csv("messages.csv")
# load categories dataset
categories = pd.read_csv("categories.csv")

#Merge datasets
df = messages.merge(categories, how='inner', on='id')

#Transform df to clean and useful dataset for ML Pipeline
#split categories to 36 columns (There're 36 categories in total)
category_name = df.categories.str.split(";", expand=True).loc[1,:].apply(lambda x: x[:-2])
df_category = df.categories.str.split(";", expand=True)
df_category.columns = category_name
df_category = df_category.applymap(lambda x: x[-1:])
# drop the original categories column from `df`
df.drop('categories', axis=1, inplace=True)
# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df, df_category], axis=1)
# drop duplicates
df = df.drop_duplicates()

#SAVE the clean dataset into an sqlite database
engine = create_engine('sqlite:///InsertDatabaseName.db')
df.to_sql('msg_category', engine, index=False)
