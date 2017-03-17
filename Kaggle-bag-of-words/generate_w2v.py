#!/usr/bin/env python
import pandas as pd
import nltk.data
import logging
import os.path
import numpy as np

from KaggleWord2VecUtility import KaggleWord2VecUtility

from gensim.models import Word2Vec


def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0

    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])

    if nwords != 0:
        featureVec /= nwords
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    counter = 0
    
    for review in reviews:
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter = counter + 1
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews


if __name__ == '__main__':
    train  = pd.read_csv('../data/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
    #test = pd.read_csv('../data/testData.tsv', header=0, delimiter="\t", quoting=3)
    unsup = pd.read_csv('../data/unlabeledTrainData.tsv', header=0,  delimiter="\t", quoting=3 )
    
    n_dim = 5000
	
    model_name = "%dfeatures_40minwords_10context" % n_dim
    
    
    if not os.path.exists(model_name): 
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        
        sentences = []
 
        print "Parsing sentences from training set"
        for review in train["review"]:
            sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
     
        print "Parsing sentences from unlabeled set"
        for review in unsup["review"]:
            sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)


        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
                            level=logging.INFO)
 

        num_features = n_dim    # Word vector dimensionality
        min_word_count = 5   # Minimum word count
        num_workers = 4       # Number of threads to run in parallel
        context = 10          # Context window size
        downsampling = 1e-3   # Downsample setting for frequent words
 
        print "Training Word2Vec model..."
        model = Word2Vec(sentences, workers=num_workers, \
                         size=num_features, min_count = min_word_count, \
                         window = context, sample = downsampling, seed=1)
 
        model.init_sims(replace=True)
        model.save(model_name)