#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creating an article summarizer : 
    1) use your desirable url as source 
    2) parse it with lxml
    3) find all sentences in the parsed documents
    4) Preprocessing the text
    5) Creating the weighted histogram
    6) Calculating the sentence scores
    7) Choosing the best sentences 

@author: Peyman
"""

# Importing library 
import bs4 as bs
import urllib.request
import re
import nltk
nltk.download('stopwords')
import heapq

# Getting the data -> return the complete HTML document
source = urllib.request.urlopen('https://en.wikipedia.org/wiki/Natural_language_processing').read()

# Parsing the document, Note that Lxml is your parser type
soup = bs.BeautifulSoup(source, 'lxml')

# Finding the sentences in the html file -> in wilipedia all paragraphs are in the 'p' Tag while in other sites this paragraphs might be in other tags such as 'div' or 'span' or etc 
text = ""
for paragraph in soup.find_all('p'):
    text += paragraph.text
    
### Preprocessing the text 
# removing the refrences
text = re.sub(r'\[[0-9]*\]', ' ', text)
text = re.sub(r'\s+', ' ', text)
clean_text = text.lower()
clean_text = re.sub(r'\W',' ',clean_text)
clean_text = re.sub(r'\d',' ',clean_text)
clean_text = re.sub(r'\s+',' ',clean_text)

# Tokenizing the sentences -> the input must be text due to the fact that clean_text does not include '.' or ',' or etc.
sentences = nltk.sent_tokenize(text)

# In order to create a useful histogram we need to know the stop words to exclude them from our histogram
stop_word = nltk.corpus.stopwords.words('english')

# Creating the Histogram
word2count = {}
for word in nltk.word_tokenize(clean_text):
    if word not in stop_word:
        if word not in word2count:
            word2count[word] = 1
        else: 
            word2count[word] += 1

# Creating the weighted histogram  -> we divide each value to maximum value in the dictionary
for key in word2count.keys():
    word2count[key] = word2count[key]/max(word2count.values())    
        
# Calculating the sentence scores
sent2score = {}
for sentence in sentences:
    # adding a filter to check if a sentence is too long do not consider it as a beneficial sentence
    if len(sentence.split(' ')) < 30:
        # sentence is not lower due to the fact that we use text in order to create sentences
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word2count.keys():
                if sentence not in sent2score:
                    sent2score[sentence] = word2count[word]
                else:
                    sent2score[sentence] += word2count[word]

# Choosing the best sentences for summary
best_sentences = heapq.nlargest(2, sent2score, key=sent2score.get)
        
print('\n********************** summary ***************************\n')
for sentence in best_sentences:
    print(sentence)    
        
        
        
        
        
        
        
        
        