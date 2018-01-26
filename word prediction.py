# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 08:55:51 2017

@author: Mirith

Info: using a neural network to predict the language of a given word, given a couple hundred translations of a document.

"""

# dataset
from nltk.corpus import udhr
# regex
import re
# to split dataset
from sklearn.model_selection import train_test_split
# machine learning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
# to save trained network
from sklearn.externals import joblib


###############################################
# presumably chunking data into trigrams by language name with regex
# this was not written by me
encoding = r"-([^-]+)$"
triplets = set([])
encodings = []

for f in udhr.fileids():
    lang = re.sub(encoding,"",f)
    enco = re.findall(encoding,f)
    if len(enco)>0:
        triplets |= set([(w,lang,enco[0]) for w in udhr.words(f)])
words,langs,encos = zip(*triplets)

words,langs,encos = zip(*[t for t in triplets\
                          if t[2] in [u"UTF8",u"Latin1",u"Latin2"]])

###############################################
# splitting data
# 
# words_train -- training input data
# words_test -- testing input data
# langs_train -- training input results
# langs_test -- testing input results
words_train, words_test, langs_train, langs_test = train_test_split(words, langs, test_size = 0.50, random_state = 48)

###############################################
# making a neural network
nnet = MLPClassifier(hidden_layer_sizes = (), 
                    learning_rate_init = .1, # learning rate
                    activation = 'logistic', # activation function
                    max_iter = 3, # number of epochs
                    alpha = 1e-4, # ????????????
                    solver = 'adam', # works well for large datasets
                    verbose = True, # changes output format
                    tol = 1e-4, # minimum tolerance for improvement
                    random_state = 1)

text_pred = Pipeline([('vect', CountVectorizer(analyzer = 'char', ngram_range = (1, 4))),
                    ('tfidf', TfidfTransformer()),
                    ('clf', nnet)])

###############################################
# training
text_pred = text_pred.fit(words_train, langs_train)

###############################################
# testing
from sklearn.metrics import accuracy_score
pred_test = text_pred.predict(words_test)
print("testing set score:", accuracy_score(langs_test, pred_test))

###############################################
# dumping trained network/saving it for later
# use whatever path you want to save to, not the one here
dump_dir = "C:/Users/________/Documents/text_pred.pkl"
joblib.dump(text_pred, dump_dir)
