"""
Utility script for infer notebook
"""
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Lambda
from tensorflow import expand_dims
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Flatten
import tensorflow

import pandas as pd
import random
import string
import numpy as np
import pickle
import os
# import matplotlib.pyplot as plt

# cleaning utilities 

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer
from html.parser import HTMLParser
from nltk.corpus import stopwords 
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

from enchant import Dict
dictionary = Dict("en_US")

porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = list(set(stopwords.words('english')))
h = HTMLParser()

def lemmatize_and_stem(word) :
    """
    Function to lemmatization and stemming of a word based on dictionary checks
    """
    lemmatized = lemmatizer.lemmatize(word)
    if lemmatized != word : 
        if dictionary.check(lemmatized)==True : 
            return lemmatized
        else : 
            return porter.stem(word)
    else : 
        stemmed = porter.stem(word)
        if dictionary.check(stemmed) == True : 
            return stemmed
        else : 
            return word
        
def clean_text(text, broken_sentences=False) : 
    """
    Function to clean sentences. If broken_sentences=True break the paragraphs in text to get the single sentences
    in cleaned text

    Steps Done are
    1. html unescape 
    2. Remove Punctuations
    3. Lemmatization and Stemming
    4. Remove StopWords 
    
    """
    cleaned_text = [] 
    if type(text) != list : 
        text = [text] 
    for paragraph in text :  
        sentence_tokenized = sent_tokenize(paragraph)
        cleaned_sentences = [] 
        for t in sentence_tokenized : 
            html_escaped_chars = h.unescape(t)
            remove_punctuations = "".join([c for c in html_escaped_chars if not c in string.punctuation])
            words = remove_punctuations.split(" ")
            lemmatized_and_stemmed = [lemmatize_and_stem(word) for word in words if len(word)>0]
            stopwords_removed_words = [word for word in lemmatized_and_stemmed if not word in stop_words]
            final_sentence = " ".join(stopwords_removed_words)
            cleaned_sentences.append(final_sentence)
        if not broken_sentences : 
            cleaned_paragraph = ". ".join(cleaned_sentences)
            cleaned_text.append(cleaned_paragraph)
        else : 
            cleaned_text = cleaned_text + cleaned_sentences
    return cleaned_text

def get_word_vector(word, word2vecmodel) : 
    return word2vecmodel.wv[word]

def get_word2vec_input_matrix(list_of_words, wordvec_model) : 
    """
    List of list of words
    Each entry in list_of_words is a list of words. All entries are of the same length
    """
    store_all_together = []
    for word_group in list_of_words : 
        store_all_together.append(np.array([get_word_vector(word, wordvec_model) for word in word_group]))
    return np.array(store_all_together)
