# --------------
# Importing Necessary libraries
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score , f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the 20newsgroups dataset
newsgroups_train = fetch_20newsgroups(subset='train')

#Create a list of 4 newsgroup and fetch it using function fetch_20newsgroups
pprint(list(newsgroups_train.target_names))
categories = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train',categories=categories)
#Use TfidfVectorizer on train data and find out the Number of Non-Zero components per sample.
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
vectors.shape
print(vectors.nnz / float(vectors.shape[0]))

newsgroups_test = fetch_20newsgroups(subset='test',categories=categories)
vectors_test = vectorizer.transform(newsgroups_test.data)
clf = MultinomialNB()
clf.fit(vectors, newsgroups_train.target)
pred = clf.predict(vectors_test)
print(f1_score(newsgroups_test.target, pred, average='macro'))

def show_top10(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print("%s: %s" % (category, " ".join(feature_names[top10])))

show_top10(clf, vectorizer, newsgroups_train.target_names)




