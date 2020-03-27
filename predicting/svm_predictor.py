from nltk.tokenize import word_tokenize
import string
from sklearn import svm
import pandas as pd
import gensim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import precision_recall_fscore_support
import os
import statistics


#make this file callable functions for our application.py file
#changetest
MAX_LENGTH = 280 # max length of a tweet
CHARS = list(string.printable) #100

labelling = pd.read_csv('/Users/b2077/PycharmProjects/pleasework/TMD_HateSpeech/data sets/Data-Set 33_33_34.csv', sep='\t')
LABELS = list(labelling['1'].values)
UNIQUE_LABELS = set(LABELS)
TRAINING_TWEETS = labelling['0'].values



# #make this csv link to the twintoutput we want to classify
twintoutput = pd.read_csv('/Users/b2077/PycharmProjects/pleasework/TMD_HateSpeech/predicting/modifiedcode (2)/datafiles/evaluation_set.csv')
TWEETS = list(twintoutput['tweets'].astype(str).values)


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




def get_accuracy(predicted, true):
    matches = 0
    for i in range(len(predicted)):
        if predicted[i] == true[i]:
            matches += 1
    return matches/len(predicted)


word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format(os.path.normpath('/Users/b2077/PycharmProjects/Maincodes/oldd/properly structured/GoogleNews-vectors-negative300.bin'), binary=True,unicode_errors='ignore')

embedded_tweets, unknown_words = sentence_embedding(TRAINING_TWEETS,word_embedding_model)

scores = []
accuracies = []
for i in range(1):
    x_train, x_test, y_train, y_test = train_test_split(embedded_tweets, LABELS, test_size=0.2, random_state=666, shuffle=True)

    classifier = svm.LinearSVC(loss='hinge', C=1.0)
    classifier.fit(x_train,y_train)

    # list_of_tweets_to_predict = TWEETS
    # embedded_tweets_to_predict, unknown_words_ = sentence_embedding(list_of_tweets_to_predict, word_embedding_model)

    predicted_labels = classifier.predict(x_test) # this is the array of predictions with the labels
    # score = prfs(y_test, predicted_labels)
    # scores.append(score)
    # accuracy = get_accuracy(predicted_labels, y_test)
    # accuracies.append(accuracy)
    # print(len(predicted_labels))
    # print(accuracy)
    # print(scores)

# print('these are the accuracies', accuracies)
# print('these are the scores', scores)
# print(statistics.mean(accuracy))

actual_labels = twintoutput['label'].values

prediction_dict = { 'Predictions': predicted_labels, "True Labels": y_test


}

predictions_datafame = pd.DataFrame(prediction_dict)
predictions_datafame.to_csv("googlenpredictionsonbigset",index = False)