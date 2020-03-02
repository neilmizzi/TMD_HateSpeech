from nltk.tokenize import word_tokenize
import string
from sklearn import svm
import pandas as pd
import gensim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

MAX_LENGTH = 280 #max lenght of a tweet
CHARS = list(string.printable) #100

DATA = pd.read_csv('/Users/b2077/PycharmProjects/Maincodes/TDM/labeled_data (1).txt')
TWEETS = list(DATA['tweet'].astype(str).values)
LABELS = list(DATA['class'].values)
UNIQUE_LABELS = set(LABELS)
print(list(UNIQUE_LABELS))

print(len(UNIQUE_LABELS))
print("Hate Speech:", LABELS.count(0))
print("Offensive:", LABELS.count(1))
print("neither:", LABELS.count(2))


# def transform_tweets_into_embeddings(tweets,embedding_model):
#     embedded_tweets = []
#     unknown_words = 0
#     for tw in tweets:
#         tweet = []
#         tokens = word_tokenize(tw)
#         for token in tokens:
#             if token in embedding_model:
#                 transformed_token = embedding_model[token]
#                 tweet.append(transformed_token)
#             else:
#                 unknown_words += 1
#                 print('unknown word')
#         embedded_tweets.append(tweet)
#             # elif token not in embedding_model:
#             #     transformed_token = [0]*300
#             #     transformed_words.append(transformed_token)
#     return embedded_tweets, unknown_words

def sentence_embedding(tweets:list, embedder):
    embedded_tweets = []
    unknown_words = 0
    for tw in tweets:
        tokens_in_tweet = word_tokenize(tw)
        sentence = [] # create list for summing
        for word in tokens_in_tweet:
            if word in embedder:
                embedded_word = embedder[word]
            else:
                unknown_words += 1
                print('unknown word')
                embedded_word = [0] * 300
            sentence.append(embedded_word)

        sum = np.array(sentence).sum(axis=0)
        #
        # for x in sum:
        #     print(type(x))

        average_vector = [ elem / len(sentence) for elem in list(sum)]

        embedded_tweets.append(average_vector)
    return embedded_tweets, unknown_words

word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format('/Users/b2077/PycharmProjects/Maincodes/oldd/properly structured/GoogleNews-vectors-negative300.bin', binary=True)

embedded_tweets, unknown_words = sentence_embedding(TWEETS,word_embedding_model)
#
x_train, x_test, y_train, y_test = train_test_split(embedded_tweets, LABELS, test_size=0.2, random_state=0, shuffle = False)
#
classifier = svm.LinearSVC(loss='hinge', C=1.0)
#
classifier.fit(x_train,y_train)
# #
predicted_labels = classifier.predict(x_test)
# #
print(predicted_labels)

prec, recall, fscore, support = precision_recall_fscore_support(predicted_labels, y_test, average=None, labels=list(UNIQUE_LABELS))

dataframe = pd.DataFrame([prec,recall,fscore], index = ['Precision','Recall', 'F-Score'], columns = ['hate speech', 'offensive', 'neither'])

print(dataframe)


