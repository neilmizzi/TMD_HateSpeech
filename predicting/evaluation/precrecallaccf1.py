from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def get_accuracy(predicted, true):
    matches = 0
    for i in range(len(predicted)):
        if predicted[i] == true[i]:
            matches += 1
    return matches/len(predicted)

predictions = pd.read_csv(os.path.normpath('/Users/b2077/PycharmProjects/pleasework/TMD_HateSpeech/predicting/modifiedcode (2)/optimalsettingsXTEST_'))
true = pd.read_csv(os.path.normpath('/Users/b2077/PycharmProjects/pleasework/TMD_HateSpeech/predicting/modifiedcode (2)/optimalsettingsXTEST_'))

truee = true['True Labels'].values
predictions_ = predictions['Predictions'].values

# data = pd.read_csv(os.path.normpath('C:/Users/Guido/Documents/Master AI/Year 1/Module 4/Text Mining Domains/Code'
#                                    '/TMD_HateSpeech/predicting/svm_labels_self_annotated.csv'))
# predictions = data.loc[:, 'Predictions']
# true = data.loc[:, 'True Labels']

precision = prfs(truee, predictions_)
confusion_df = pd.DataFrame(confusion_matrix(truee, predictions_))
x_axis_labels = ['Hate', 'Offensive', 'None'] # labels for x-axis
y_axis_labels = ['Hate', 'Offensive', 'None'] # labels for y-axis
heatmap = sns.heatmap(confusion_df, annot=True, fmt='g', cmap='Blues', xticklabels=x_axis_labels, yticklabels=y_axis_labels)

print(get_accuracy(truee, predictions_))
print(confusion_df)
print(precision)

plt.show()