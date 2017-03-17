#!/usr/bin/env python
import logging
import os.path
import pandas as pd
import numpy as np

from KaggleWord2VecUtility import KaggleWord2VecUtility

from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence


def getFeatureVecs(reviews, model, num_features):
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    counter = -1
     
    for review in reviews:
        counter += 1
        try:
            reviewFeatureVecs[counter] = np.array(model[review.labels[0]]).reshape((1, num_features))
        except:
            continue
    return reviewFeatureVecs


def getCleanLabeledReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append(KaggleWord2VecUtility.review_to_wordlist(review))
    
    labelized = []
    for i, id_label in enumerate(reviews["id"]):
        labelized.append(LabeledSentence(clean_reviews[i], [id_label]))
    return labelized



if __name__ == '__main__':
    train  = pd.read_csv('../data/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
    test = pd.read_csv('../data/testData.tsv', header=0, delimiter="\t", quoting=3)
    unsup = pd.read_csv('../data/unlabeledTrainData.tsv', header=0,  delimiter="\t", quoting=3 )
   
    print "Cleaning and labeling all data sets...\n"
    
    train_reviews = getCleanLabeledReviews(train)
    test_reviews = getCleanLabeledReviews(test)
    unsup_reviews = getCleanLabeledReviews(unsup)

    n_dim =5000
	
    model_dm_name = "%dfeatures_1minwords_10context_dm" % n_dim
    model_dbow_name = "%dfeatures_1minwords_10context_dbow" % n_dim
    
    
    
    if not os.path.exists(model_dm_name) or not os.path.exists(model_dbow_name):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
                            level=logging.INFO)
 
        num_features = n_dim    # Word vector dimensionality
        min_word_count = 1   # Minimum word count, if bigger, some sentences may be missing
        num_workers = 4       # Number of threads to run in parallel
        context = 10          # Context window size
        downsampling = 1e-3   # Downsample setting for frequent words
 
        print "Training Doc2Vec model..."
        model_dm = Doc2Vec(min_count=min_word_count, window=context, size=num_features, \
                           sample=downsampling, workers=num_workers)
        model_dbow = Doc2Vec(min_count=min_word_count, window=context, size=num_features, 
                             sample=downsampling, workers=num_workers, dm=0)
        
        all_reviews = np.concatenate((train_reviews, test_reviews, unsup_reviews))
        model_dm.build_vocab(all_reviews)
        model_dbow.build_vocab(all_reviews)
        
        for epoch in range(10):
            perm = np.random.permutation(all_reviews.shape[0])
            model_dm.train(all_reviews[perm])
            model_dbow.train(all_reviews[perm])
            
        model_dm.save(model_dm_name)
        model_dbow.save(model_dbow_name)
