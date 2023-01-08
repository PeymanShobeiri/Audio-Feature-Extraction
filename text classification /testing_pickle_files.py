#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing the trained pickle files 

@author: Peyman
"""
import pickle

# Unpickling the classifier and vectorizer 
with open('classifier.pickle', 'rb') as f:
    clf = pickle.load(f)

with open('tfidfmodel.pickle', 'rb') as f:
    tfidf = pickle.load(f)
    
# Testing
sample_sent = ["you are a nice man bro"]
sample_sent = tfidf.transform(sample_sent).toarray()

# predicting the result -> 0 means - and 1 means +
print(clf.predict(sample_sent))