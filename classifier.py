#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re, random, math, collections, itertools, pdb

# Gives the emotions we are interested in classifying.  To classify an
# additional emotion, add its name here and an accompanying data file with some
# training data.
def SENTIMENTS():
    return ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised']

# The Bayes classifier gives a number between 0 and 1, expressing the
# confidence that the sentence is of a certain sentiment. The threshold
# determines the required certainty for a sentence to be classified as that
# sentiment.
def THRESHOLD():
    return 1.0 / float(len(SENTIMENTS()))

#  Adapted from Scott Triglia. Elegant N-gram Generation in Python. (2013-01-20).
#  URL: http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/.
#  Accessed: 2013-11-21.
#  Archived by WebCite® at http://www.webcitation.org/6LImC39zN
def makeNgram(wordList, order):
    ngramList = []
    for ngram in zip(*[wordList[i:] for i in range(order)]):
        ngramList.append('_'.join(ngram))
    return ngramList

def readFiles(sentencesTrain,sentencesTest):
    for sentiment in SENTIMENTS():
        txt = open('emotions/' + sentiment + '.txt')
        for sentence in re.split(r'\n', txt.read()):
            if random.randint(1,10)<2:
                sentencesTest[sentence] = sentiment
            else:
                sentencesTrain[sentence] = sentiment

#calculates p(W|Positive), p(W|Negative) and p(W) for all words in training data
def trainBayes(sentencesTrain, pWord):

    freq = {}
    for sentiment in SENTIMENTS():
        freq[sentiment] = {}

    dictionary = {}

    wordTotals = { 'all': 0 }
    for sentiment in SENTIMENTS():
        wordTotals[sentiment] = 0

    #iterate through each sentence/sentiment pair in the training data
    for sentence, sentiment in sentencesTrain.iteritems():
        if sentence == '':
            continue

        wordList = re.findall(r"[\w']+", sentence)

        unigramList = wordList
        bigramList = makeNgram(wordList, 2)
        trigramList = makeNgram(wordList, 3)

        for word in (trigramList + bigramList + unigramList):
            wordTotals['all'] += 1
            if not dictionary.has_key(word):
                dictionary[word] = 1
            wordTotals[sentiment] += 1
            if not freq[sentiment].has_key(word):
                freq[sentiment][word] = 1
            else:
                freq[sentiment][word] += 1


    # smoothing so min count of each word is 1
    for word in dictionary:
        for sentiment in SENTIMENTS():
            if not freq[sentiment].has_key(word):
                freq[sentiment][word] = 1
            else:
                freq[sentiment][word] += 1

        # divisor for the p(word) calculation
        freqWordAll = 0
        for sentiment in SENTIMENTS():
            freqWordAll += freq[sentiment][word]

        # p(word|sentiment)
        for sentiment in SENTIMENTS():
            pWord[sentiment][word] = freq[sentiment][word] / float(wordTotals[sentiment])

        #p(word)
        pWord['all'][word] = freqWordAll / float(wordTotals['all']) 


#INPUTS:
#  sentences is a dictonary of { sentence: sentiment } for every sentence.
#  pWord is dictionary storing p(word) and p(word|sentiment)
def testBayes(sentences, pWord):
    total = {}
    correct = {}
    total['all'] = 0
    correct['all'] = 0
    for sentiment in SENTIMENTS():
        total[sentiment] = 0
        correct[sentiment] = 0

    #for each sentence, sentiment pair in the dataset
    for sentence, sentiment in sentences.iteritems():
        if sentence == '':
            continue

        wordList = re.findall(r"[\w']+", sentence)#collect all words

        unigramList = []
        bigramList = []
        trigramList = []

        unigramList = wordList

        if len(wordList) > 1:
            bigramList = ["<sen>_" + wordList[0]]
        bigramList = bigramList + makeNgram(wordList, 2)
        if len(wordList) > 1:
            bigramList = bigramList + [wordList[-1] + "_</sen>"]

        if len(wordList) > 2:
            trigramList = trigramList + ["<sen>_<sen>_" + wordList[0]]
        if len(wordList) > 1:
            trigramList = trigramList + ["<sen>_" + wordList[0] + "_" + wordList[1]]
        trigramList = trigramList + makeNgram(wordList, 3)
        if len(wordList) > 2:
            trigramList = trigramList + [wordList[-2] + "_" + wordList[-1] + "_</sen>"]
        if len(wordList) > 1:
            trigramList = trigramList + [wordList[-1] + "_</sen>_</sen>"]

        p = {}
        for s in SENTIMENTS():
            p[s] = THRESHOLD()

        for word in (trigramList + bigramList + unigramList):
            if pWord['all'].has_key(word):
                for s in SENTIMENTS():
                    if pWord[s][word] > 0.00000001:
                        p[s] *= pWord[s][word] * 10000

        total['all'] += 1
        total[sentiment] += 1

        totalProb = float(sum(p.itervalues()))
        prob = 0
        prob = p[sentiment] / totalProb
        if prob >= THRESHOLD():
            correct['all'] += 1
            correct[sentiment] += 1

    accuracy = {}
    accuracy['all'] = correct['all'] / float(total['all'])

    print " (ALL)=%0.2f" % accuracy['all'] + " (%d" % correct['all'] + "/%d" % total['all'] + ")"
    for sentiment in SENTIMENTS():
        accuracy[sentiment] = correct[sentiment] /  float(total[sentiment])
        print " (" + sentiment + ")=%0.2f" % accuracy[sentiment] + " (%d" % correct[sentiment] + "/%d" % total[sentiment] + ")"

#---------- Main Script --------------------------


#initialise datasets and dictionaries
sentencesTrain={}
sentencesTest={}
pWord={ 'all': {} }
for sentiment in SENTIMENTS():
    pWord[sentiment] = {}

# split the sentiment files into training and test sets
readFiles(sentencesTrain,sentencesTest)

#build conditional probabilities using training data
trainBayes(sentencesTrain, pWord)

#run naive bayes classifier on datasets
print "Naive Bayes"
print "Train Accuracy"
testBayes(sentencesTrain, pWord)
print "Test Accuracy"
testBayes(sentencesTest, pWord)
