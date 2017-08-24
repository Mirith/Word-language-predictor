# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 08:55:51 2017

@author: Mirith

Info: given a trained network that predicts the language of a word, this will use that information to plot language similarity to English.  

"""

###############################################
# loading a previously trained and saved analyzer
text_pred = joblib.load("C:/Users/_________/Documents/text_pred.pkl")

# testing accuracy
from sklearn import metrics
print(metrics.classification_report(langs_test, pred_test,
target_names = text_pred.classes_))

###############################################
# plotting language similarities
import numpy_indexed as npi
pred_prob = text_pred.predict_proba(words_test)
avgs = npi.group_by(langs_test).mean(pred_prob)

print(avgs[1].shape)

# used for plot
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from numpy.linalg import norm

# from internet, can't find source again
def JSD(P, Q):
    _P = (P / norm(P, ord = 1))
    _Q = (Q / norm(Q, ord = 1))
    _M = (.5 * (_P + _Q))
    return (1 - .5 * (entropy(_P, _M) + entropy(_Q, _M)))
similarities = squareform( pdist(avgs[1], metric = JSD) )

from numpy import where
from collections import defaultdict

langNumbers = defaultdict(float)
languages = []

# list of all languages
for language in text_pred.classes_:
    languages += [language]

# assigning languages numbers based on similarity to English
for language in languages:
    langNumbers[language] = float("{0:.2f}".format(similarities[where(text_pred.classes_ == "English")[0][0], 
    where(text_pred.classes_ == language)[0][0]] * 100))
print(langNumbers)

# sorting languages based on similarity
# from https://stackoverflow.com/questions/613183/sort-a-python-dictionary-by-value, answer by Nas Banov
sortedLang = [(k, langNumbers[k]) for k in sorted(langNumbers, key=langNumbers.get, reverse=True)]

# top 20 most similar
simLang = sortedLang[:20]

# bottom 20 least similar
# to get rid of English being 0, excluding the bottom one
disLang = sortedLang[-21:-1]

# imports for plots
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
# similar languages
langs = [lang for lang, num in simLang]
y_pos = np.arange(len(langs))
similarity = [num for lang, num in simLang]
 
plt.barh(y_pos, similarity, align='center', alpha=0.5)
plt.yticks(y_pos, langs)
plt.xlabel('Percentage')
plt.title('Calculated Similarity to English (highest)')
 
plt.show()

# dissimilar languages
langs = [lang for lang, num in disLang]
y_pos = np.arange(len(langs))
dissim = [num for lang, num in disLang]
 
plt.barh(y_pos, dissim, align='center', alpha=0.5)
plt.yticks(y_pos, langs)
plt.xlabel('Percentage')
plt.title('Calculated Similarity to English (lowest)')
 
plt.show()