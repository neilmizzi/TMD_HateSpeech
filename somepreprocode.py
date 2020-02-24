import preprocessor as p
import pandas as pd

def clean_dataframe_with_tweets(dataframe,column_of_text):
        dataframe.replace(':'," ", inplace = True, regex = True)
        dataframe.replace('\t', " ", inplace=True, regex=True)
        dataframe[column_of_text] = [p.clean(elem) for elem in dataframe[column_of_text].values]
        dataframe[column_of_text] = [elem.strip() for elem in dataframe[column_of_text].values]
        return dataframe
