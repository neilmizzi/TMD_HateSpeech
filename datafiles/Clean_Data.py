import pandas as pd
import string
from tqdm import tqdm

df = pd.read_csv('./data/Data-Set 10_40_50.csv', sep='\t')

for index, row in tqdm(df.iterrows()):
    tweet = row[1]
    for char in tweet[:]:
        if char not in list(string.printable):
            tweet = tweet.replace(char, '')
            df.iloc[index, 1] = tweet

df = df[['0','1']]
df.to_csv('./data/clean 10_40_50.csv', index=False)