import os
import pandas as pd
from langdetect import detect
from tqdm import tqdm


def language_detection(df):
    return pd.DataFrame.from_records([tweet for tweet in tqdm(df.values.tolist()) if detect(tweet[1]) == 'en'])


def process_file(path):
    df = pd.read_csv(path)
    if path == file_path_1:
        df = df[['Class', 'Tweet']]
        df.columns = ['class', 'tweet']
        hate = df.loc[df['class'] == '0']
        offensive = df.loc[df['class'] == '1']
        neither = df.loc[df['class'] == '2']
        balanced_1 = pd.concat([hate, offensive[:len(hate)], neither[:len(hate)]])
        balanced_1 = language_detection(balanced_1)
        return balanced_1
    else:
        hate = df.loc[df['class'] == 0]
        offensive = df.loc[df['class'] == 1]
        neither = df.loc[df['class'] == 2]
    balanced = pd.concat([hate, offensive[:len(hate)], neither[:len(hate)]])
    return balanced[['class', 'tweet']]


file_path_1 = os.path.normpath('./data/hate-speech/labels.csv')
file_path_2 = os.path.normpath('./data/hateoffensive-speech-detection/hate_speech_data_train.csv')
file_path_3 = os.path.normpath('./data/labeled_data.csv')

paths = [file_path_1, file_path_2, file_path_3]

data_set = pd.concat([process_file(path) for path in paths])
data_set = data_set[['class', 'tweet']]
print(data_set[130:])