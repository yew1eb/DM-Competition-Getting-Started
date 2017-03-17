import numpy as np
import pandas as pd
from collections import Counter

from KaggleWord2VecUtility import KaggleWord2VecUtility

def tokenize(sentence, grams):
    words = KaggleWord2VecUtility.review_to_wordlist(sentence)
    tokens = []
    for gram in grams:
        for i in range(len(words) - gram + 1):
            tokens += ["_*_".join(words[i:i+gram])]
    return tokens


def build_dict(data, grams):
    dic = Counter()
    for token_list in data:
        dic.update(token_list)
    return dic


def compute_ratio(poscounts, negcounts, alpha=1):
    alltokens = list(set(poscounts.keys() + negcounts.keys()))
    dic = dict((t, i) for i, t in enumerate(alltokens))
    d = len(dic)
    
    print "Computing r...\n"
    
    p, q = np.ones(d) * alpha , np.ones(d) * alpha
    for t in alltokens:
        p[dic[t]] += poscounts[t]
        q[dic[t]] += negcounts[t]
    p /= abs(p).sum()
    q /= abs(q).sum()
    r = np.log(p/q)
    return dic, r


def generate_svmlight_content(data, dic, r, grams):
    output = []
    for _, row in data.iterrows():
        tokens = tokenize(row['review'], grams)
        indexes = []
        for t in tokens:
            try:
                indexes += [dic[t]]
            except KeyError:
                pass
        indexes = list(set(indexes))
        indexes.sort()
        if 'sentiment' in row:
            line = [str(row['sentiment'])]
        else:
            line = ['0']
        for i in indexes:
            line += ["%i:%f" % (i + 1, r[i])]
        output += [" ".join(line)]
            
    return "\n".join(output)
    
    
def generate_svmlight_files(train, test, grams, outfn):
    ngram = [int(i) for i in grams]
    ptrain = []
    ntrain = []
    
    print "Parsing training data...\n"
    
    for _, row in train.iterrows():
        if row['sentiment'] == 1:
            ptrain.append(tokenize(row['review'], ngram))
        elif row['sentiment'] == 0:
            ntrain.append(tokenize(row['review'], ngram))
    
    pos_counts = build_dict(ptrain, ngram)
    neg_counts = build_dict(ntrain, ngram)
    
    dic, r = compute_ratio(pos_counts, neg_counts)
    
    f = open(outfn + '-train.txt', "w")
    f.writelines(generate_svmlight_content(train, dic, r, ngram))
    f.close()

    print "Parsing test data...\n"
    
    f = open(outfn + '-test.txt', "w")
    f.writelines(generate_svmlight_content(test, dic, r, ngram))
    f.close()
    
    print "SVMlight files have been generated!"
    
    
    