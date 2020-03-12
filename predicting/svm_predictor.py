from nltk.tokenize import word_tokenize
import string
from sklearn import svm
import pandas as pd
import gensim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

#make this file callable functions for our application.py file
#changetest
MAX_LENGTH = 280 # max length of a tweet
CHARS = list(string.printable) #100

labelling = pd.read_csv('modifiedcode (2)/labeled_data (1).txt')
LABELS = list(labelling['class'].values)
UNIQUE_LABELS = set(LABELS)
TRAINING_TWEETS = labelling['tweet'].values

#make this csv link to the twintoutput we want to classify
twintoutput = pd.read_csv('modifiedcode (2)/trump.csv')
TWEETS = list(twintoutput['tweet'].astype(str).values)



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
                embedded_word = [0] * 300
            sentence.append(embedded_word)

        sum = np.array(sentence).sum(axis=0)

        average_vector = [ elem / len(sentence) for elem in list(sum)]

        embedded_tweets.append(average_vector)
    return embedded_tweets, unknown_words

word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format('modifiedcode (2)/GoogleNews-vectors-negative300.bin', binary=True)

embedded_tweets, unknown_words = sentence_embedding(TRAINING_TWEETS,word_embedding_model)

x_train, x_test, y_train, y_test = train_test_split(embedded_tweets, LABELS, test_size=0.2, random_state=0, shuffle = False)

classifier = svm.LinearSVC(loss='hinge', C=1.0)

classifier.fit(x_train,y_train)

list_of_tweets_to_predict = TWEETS
embedded_tweets_to_predict, unknown_words_ = sentence_embedding(list_of_tweets_to_predict,word_embedding_model)

predicted_labels = classifier.predict(embedded_tweets_to_predict) # this is the array of predictions with the labels

print(len(predicted_labels))
# only relevant when training with labeled input
# prec, recall, fscore, support = precision_recall_fscore_support(predicted_labels, y_test, average=None, labels=list(UNIQUE_LABELS))
#
# dataframe = pd.DataFrame([prec,recall,fscore], index = ['Precision','Recall', 'F-Score'], columns = ['hate speech', 'offensive', 'neither'])
#
# print(dataframe)


