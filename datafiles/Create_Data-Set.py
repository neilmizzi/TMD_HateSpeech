import os
import pandas as pd
from langdetect import detect
from tqdm import tqdm
import string


# need to sum to 100
HATE_RATIO = 33
OFFENSIVE_RATI0 = 33
NEITHER_RATIO = 34


class DataSet():
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.df.columns = map(str.lower, self.df.columns)
        pd.to_numeric(self.df.loc[:, 'class'])
        self.df = self.df[['tweet', 'class']]

    def get_split(self, hate_ratio, offensive_ratio, neither_ratio):
        hate = self.df.loc[self.df['class'] == 0]
        offensive = self.df.loc[self.df['class'] == 1]
        neither = self.df.loc[self.df['class'] == 2]
        smallest_set = data_set.get_smallest_set(hate, offensive, neither)
        one_percent = int(data_set.get_one_percent(smallest_set, hate, offensive, neither))
        data_dist = pd.concat([hate[:(one_percent * HATE_RATIO)], offensive[: (one_percent * OFFENSIVE_RATI0)],
                              neither[: (one_percent * NEITHER_RATIO)]])
        return data_set.language_detection(data_dist)

    def get_smallest_set(self, hate, offensive, neither):
        subsets = [hate, offensive, neither]
        lengths = list(map(len, subsets))
        index_shortest = lengths.index(min(lengths))
        return subsets[index_shortest]

    def get_one_percent(self, set, hate, offensive, neither):
        if len(set) == len(hate): one_percent = len(set) / HATE_RATIO
        elif len(set) == len(offensive): one_percent = len(set) / OFFENSIVE_RATI0
        elif len(set) == len(neither): one_percent = len(set) / NEITHER_RATIO
        return one_percent

    def language_detection(self, data_set):
        return pd.DataFrame.from_records([tweet for tweet in tqdm(data_set.values.tolist()) if detect(tweet[0]) == 'en'])


file_path_1 = os.path.normpath('./data/hate-speech/labels.csv')
file_path_2 = os.path.normpath('./data/hateoffensive-speech-detection/hate_speech_data_train.csv')
file_path_3 = os.path.normpath('./data/labeled_data.csv')

paths = [file_path_1, file_path_2, file_path_3]
data_set_list = []

for path in paths:
    print(f'Splitting the data from {path}')
    data_set = DataSet(path)
    new_dist = data_set.get_split(HATE_RATIO, OFFENSIVE_RATI0, NEITHER_RATIO)
    data_set_list.append(new_dist)


final_data_set = pd.concat(data_set_list)
# final_data_set = final_data_set.sample(n=20210)
print(len(final_data_set))
final_data_set.to_csv('./data/Data-Set 33_33_34.csv', sep='\t')
