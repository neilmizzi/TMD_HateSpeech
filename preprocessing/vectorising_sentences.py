import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import gensim

def load_embedder(filepath):
    return gensim.models.KeyedVectors.load_word2vec_format(filepath, binary=True)

def sentence_embedding(sentence, embedder):
    """
    sentence : str
    Return the average embedding from the token embeddings in the sentence.
    """
    doc = word_tokenize(sentence)
    concatenated = [] # concatenated embeddings for each token

    for token in doc:
        try:
            word_vector = np.array(embedder[str(token)]).reshape(1, -1)
            concatenated.append(word_vector[0])
        except:
            continue

    df = pd.DataFrame(concatenated)
    average_vector = df.mean().tolist()

    if average_vector == []:
        sentence_vector = [0]*300
    else:
        sentence_vector = average_vector

    return sentence_vector

def embedd_all_sentences(sentences: list,embedder):
    """sentences: list
        embedder: loaded word embedding model"""
    embedded_sentences = []
    for sentence in sentences:
        sentence_vector = sentence_embedding(sentence,embedder) # vectorise sentence
        embedded_sentences.append(sentence_vector)
    return embedded_sentences

word_embedder_file_path = '/Users/b2077/PycharmProjects/Maincodes/properly structured/GoogleNews-vectors-negative300.bin' #paste filepath to embedder here.
word_embedder = load_embedder(word_embedder_file_path)




