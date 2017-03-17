#!/usr/bin/env python
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import scale
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_svmlight_files
from scipy.sparse import hstack

from gensim.models import Doc2Vec, Word2Vec
from gensim.models.doc2vec import LabeledSentence

from nbsvm import generate_svmlight_files

from KaggleWord2VecUtility import KaggleWord2VecUtility


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
    counter = 0

    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")

    for review in reviews:
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter = counter + 1
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append(KaggleWord2VecUtility.review_to_wordlist(review, True))
    return clean_reviews 


def getFeatureVecs(reviews, model, num_features):
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    counter = -1
    
    for review in reviews:
        counter = counter + 1
        try:
            reviewFeatureVecs[counter] = np.array(model[review.labels[0]]).reshape((1, num_features))
        except:
            continue
    return reviewFeatureVecs


def getCleanLabeledReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append(KaggleWord2VecUtility.review_to_wordlist(review, True))
    
    labelized = []
    for i, id_label in enumerate(reviews["id"]):
        labelized.append(LabeledSentence(clean_reviews[i], [id_label]))
    return labelized


if __name__ == '__main__':
    train = pd.read_csv('../data/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
    test = pd.read_csv('../data/testData.tsv', header=0, delimiter="\t", quoting=3 )
   
    print "Cleaning and parsing the data sets...\n"

    clean_train_reviews = []
    for review in train['review']:
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(review)))

    clean_test_reviews = []
    for review in test['review']:
        clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(review)))

    print "Creating the bag of words...\n"

    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,3), sublinear_tf=True)
    
    X_train_bow = vectorizer.fit_transform(clean_train_reviews)
    X_test_bow = vectorizer.transform(clean_test_reviews)
    
       
    print "Cleaning and labeling the data sets...\n"
    
    train_reviews = getCleanLabeledReviews(train)
    test_reviews = getCleanLabeledReviews(test)

    n_dim = 5000
    
    print 'Loading doc2vec model..\n'
    
    model_dm_name = "../data/%dfeatures_1minwords_10context_dm" % n_dim
    model_dbow_name = "../data/%dfeatures_1minwords_10context_dbow" % n_dim
          
    model_dm = Doc2Vec.load(model_dm_name)
    model_dbow = Doc2Vec.load(model_dbow_name)
        
    print "Creating the d2v vectors...\n"

    X_train_d2v_dm = getFeatureVecs(train_reviews, model_dm, n_dim)
    X_train_d2v_dbow = getFeatureVecs(train_reviews, model_dbow, n_dim)
    X_train_d2v = np.hstack((X_train_d2v_dm, X_train_d2v_dbow))

    X_test_d2v_dm = getFeatureVecs(test_reviews, model_dm, n_dim)
    X_test_d2v_dbow = getFeatureVecs(test_reviews, model_dbow, n_dim)
    X_test_d2v = np.hstack((X_test_d2v_dm, X_test_d2v_dbow))
    
    
    print 'Loading word2vec model..\n'
    
    model_name = "../data/%dfeatures_40minwords_10context" % n_dim
	
    model = Word2Vec.load(model_name)
    
    print "Creating the w2v vectors...\n"

    X_train_w2v = scale(getAvgFeatureVecs(getCleanReviews(train), model, n_dim))
    X_test_w2v = scale(getAvgFeatureVecs(getCleanReviews(test), model, n_dim))
    
    print "Generating the svmlight-format files...\n"
    
    generate_svmlight_files(train, test, '123', '../data/nbsvm')
    
    print "Creating the nbsvm...\n"
    
    files = ("../data/nbsvm-train.txt", "../data/nbsvm-test.txt")
     
    X_train_nbsvm, _, X_test_nbsvm, _ = load_svmlight_files(files)
    
    print "Combing the bag of words and the w2v vectors...\n"
    
    X_train_bwv = hstack([X_train_bow, X_train_w2v])
    X_test_bwv = hstack([X_test_bow, X_test_w2v])

    
    print "Combing the bag of words and the d2v vectors...\n"
    
    X_train_bdv = hstack([X_train_bow, X_train_d2v])
    X_test_bdv = hstack([X_test_bow, X_test_d2v])

    
    print "Checking the dimension of training vectors"
    
    print 'BoW', X_train_bow.shape
    print 'W2V', X_train_w2v.shape
    print 'D2V', X_train_d2v.shape
    print 'NBSVM', X_train_nbsvm.shape
    print 'BoW-W2V', X_train_bwv.shape
    print 'BoW-D2V', X_train_bdv.shape
    print ''

    y_train = train['sentiment']
    
    
    print "Predicting with Bag-of-words model...\n" 
    
    clf = LogisticRegression(class_weight="auto")
    
    clf.fit(X_train_bow, y_train)
    y_prob_bow = clf.predict_proba(X_test_bow)
    
    print "Predicting with NBSVM...\n" 
	
    clf.fit(X_train_nbsvm, y_train)
    y_prob_nbsvm = clf.predict_proba(X_test_nbsvm)
	
	
	print "Predicting with Bag-of-words model and Word2Vec model...\n" 
	
    clf.fit(X_train_bwv, y_train)
    y_prob_bwv = clf.predict_proba(X_test_bwv)
    
	
	print "Predicting with Bag-of-words model and Doc2Vec model...\n" 
	
    clf.fit(X_train_bdv, y_train)
    y_prob_bdv = clf.predict_proba(X_test_bdv)
    

    print "\nWeighted Average: BOW/BOW-W2V/BOW-D2V/NBSVM\n"
    
    alpha = 0.081633
    beta = 0.265306
    theta = 0.551020
      
    y_pred = alpha*y_prob_bow + (1-alpha-beta-theta)*y_prob_bwv + beta*y_prob_bdv + theta*y_prob_nbsvm

    output = pd.DataFrame(data={"id":test["id"], "sentiment":y_pred[:,1]})
    output.to_csv('BoW008_W2V5000_D2V10000_NBSVM055_model.csv', index=False, quoting=3)
    
    print "Wrote results to BoW008_W2V5000_D2V10000_NBSVM055_model.csv"   
    

    print "\nMax-Min (Average)\n"
    y_mean = (y_prob_bow + y_prob_bwv + y_prob_bdv + y_prob_nbsvm)/4
    y_score_mean = []
    
    i = 0
    for row in y_mean:
        if row[1] > 0.5:
            val = max(y_prob_bow[i,1],y_prob_bwv[i,1],y_prob_bdv[i,1],y_prob_nbsvm[i,1])
            y_score_mean.append(val)
        elif row[1] < 0.5:
            val = min(y_prob_bow[i,1],y_prob_bwv[i,1],y_prob_bdv[i,1],y_prob_nbsvm[i,1])
            y_score_mean.append(val)
        else:
            y_score_mean.append(y_pred[i,1])
        i += 1
    
 
    print "\nMax-Min (Weighted Average)\n"
    y_score_best = []
    
    i = 0
    for row in y_pred:
        if row[1] > 0.5:
            val = max(y_prob_bow[i,1],y_prob_bwv[i,1],y_prob_bdv[i,1],y_prob_nbsvm[i,1])
            y_score_best.append(val)
        elif row[1] < 0.5:
            val = min(y_prob_bow[i,1],y_prob_bwv[i,1],y_prob_bdv[i,1],y_prob_nbsvm[i,1])
            y_score_best.append(val)
        else:
            y_score_best.append(y_pred[i,1])
        i += 1
 
 
    print "\nFinal Ensemble\n"
    y_wa = np.array([row[1] for row in y_pred])
    y_am = np.array(y_score_mean)
    y_wam = np.array(y_score_best)
    
    alpha1 = 0.591837
    alpha2 = 0.387755
    y_final = alpha1*y_wa + (1-alpha1-alpha2)*y_am + alpha2*y_wam

    output = pd.DataFrame(data={"id":test["id"], "sentiment":y_final})
    output.to_csv('WeightedAverage059_MaxMinAverage_MaxMinWeightedAverage039_model.csv', index=False, quoting=3)
    
    print "Wrote results to WeightedAverage059_MaxMinAverage_MaxMinWeightedAverage039_model.csv"
    