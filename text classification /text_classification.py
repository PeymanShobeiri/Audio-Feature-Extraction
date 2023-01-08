#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Classification project

@author: Peyman
"""

# Importing library
import numpy as np 
import nltk
import re
import pickle
from nltk.corpus import stopwords
from sklearn.datasets import load_files
# nltk.download('stopwords')

# Importing Dataset
# reviews contain all the files in the input file
reviews = load_files('txt_sentoken/')

# x contain all the  diffrent reviews and y contains their classes 
# 0 -> Negative         1 -> Positive
x,y = reviews.data, reviews.target

# Storing as pickle files -> you can save a lot of time after importing the file for the first time 
# due to the fact that you can load your objects from your pickle files
with open("x.pickle",'wb') as f:
    pickle.dump(x,f)
    
with open("y.pickle", 'wb') as f:
    pickle.dump(y,f)
    
# unpickling the Dataset
with open('x.pickle', 'rb') as f:
    x = pickle.load(f)
    
with open('y.pickle', 'rb') as f:
    y = pickle.load(f)
    
# Preprocess -> creating the corpus
# Note: type of x is bytes
corpus = []
for i in range(0,len(x)):
    review = re.sub(r'\W'," ", str(x[i]))
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+' , ' ', review)
    review = re.sub(r'^[a-z]\s+', ' ', review)
    review = re.sub(r'\s+', ' ', review)
    corpus.append(review)

# Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
# max_features -> the number of words that we select from histogram -> here: top 2000 words are selected
# min_df -> if a word appear min_df times or less than that we exclude it -> here specified as number which is 3 times
# max_df -> exlude all the words that appear in the max_df persent or times -> here : 0.6 means 60% for example we exclude the commen words such as the etc
# stop_words -> excludeing the stop words 

vectorizer = CountVectorizer(max_features=2000, min_df=3, max_df=0.6, stop_words=stopwords.words('english'))

# forming the BOW -> as an array
# x have 2000 rows which are 2000 documents and it have 2000 columns which are top 2000 words
x = vectorizer.fit_transform(corpus).toarray()

# Transform BOW model into TF-IDF Model
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
x = transformer.fit_transform(x).toarray()


# this part is for pickling part only due to the fact that we use the BOW model in this project and then transform it to tfidf model
#from sklearn.feature_extraction.text import TfidfVectorizer
#vectorizer = TfidfVectorizer(max_features=2000, min_df=3, max_df=0.6, stop_words=stopwords.words('english'))
#x = vectorizer.fit_transform(corpus).toarray()


# Creating training and test set -> we need to split the data to the test and train set 
from sklearn.model_selection import train_test_split
# x -> TF_IDF model
# y -> Matrix of classes for each review (- or +)
# test_size -> persent of data that need to be use as test 
# random_state -> Controls the shuffling applied to the data before applying the split.
text_train, text_test, send_train, send_test = train_test_split(x,y,test_size=0.2,random_state=0)

# Training our classifier -> we use logistic regression -> y = a + bx1 + cx2 + ... + zx2000 -> finds out each review is - or +
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(text_train, send_train)

# Testing our model -> checking if our model could predict correctly
# send_pred contian the 0 and 1 which are the real results
send_pred = classifier.predict(text_test)

# In order to see the result better we use confusion_matrix
from sklearn.metrics import confusion_matrix
# confusion_matrix -> retrn a matrix that help to compare the results better
# in cm -> 168 and 171 is the correct gusses and 21 and 40 are the wrong ones
# So the total accuracy is : 339/400 = 84.75
cm = confusion_matrix(send_test, send_pred)


# Pickling the classifier for future use due to the fact that it uses less time this way
with open('classifier.pickle','wb') as f:
    pickle.dump(classifier, f)

# Pickling the vectorizer 
with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer, f)
















