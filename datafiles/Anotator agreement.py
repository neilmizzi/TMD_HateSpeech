import pandas as pd
import glob
import os
import numpy as np
import itertools
from tqdm import tqdm

tweets = pd.DataFrame()
labels = pd.DataFrame()
no_major = pd.DataFrame()

for idx, file in enumerate(glob.glob(os.path.normpath('./data/Evaluation/*.csv'))):
    df = pd.read_csv(file, encoding='ISO-8859-1')
    tweets['tweets'] = df.iloc[:, 0]
    labels[str(idx)] = df.iloc[:, -1].str.lower()

labels = labels[9:]
tweets['label'] = np.nan

counter = 0
for idx, row in labels.iterrows():
    count = row.value_counts()
    if count[0] > 1:
        tweets.loc[[idx], ['label']] = count.index[0]
    else:
        t = tweets.loc[[idx], ['tweets']]
        no_major = pd.concat([no_major, t])

# no_major.to_csv('no majority.csv')
tweets.to_csv('evaluation_set.csv')