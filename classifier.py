#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re, random, math, collections, itertools, pdb

def sentiments():
    return ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised']

#  Scott Triglia. Elegant N-gram Generation in Python. (2013-01-20).
#  URL: http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/.
#  Accessed: 2013-11-21.
#  Archived by WebCite® at http://www.webcitation.org/6LImC39zN
def makeNgram(wordList, order):
    ngramList = []
    for ngram in zip(*[wordList[i:] for i in range(order)]):
        ngramList.append('_'.join(ngram))
    return ngramList

#------------ Function Definitions ---------------------

def readFiles(sentencesTrain,sentencesTest):

    #reading pre-labeled movie reviews and splitting into lines

    sentences = {}
    for sentiment in sentiments():
        txt = open('emotions/' + sentiment + '.txt')
        sentences[sentiment] = re.split(r'\n', txt.read())

    for sentiment,sentences in sentences.iteritems():
        for sentence in sentences:
            if random.randint(1,10)<2:
                sentencesTest[sentence] = sentiment
            else:
                sentencesTrain[sentence] = sentiment

#----------------------------End of data initialisation ----------------#

#calculates p(W|Positive), p(W|Negative) and p(W) for all words in training data
def trainBayes(sentencesTrain, pWord):

    freq = {}
    for sentiment in sentiments():
        freq[sentiment] = {}

    dictionary = {}

    wordTotals = { 'all': 0 }
    for sentiment in sentiments():
        wordTotals[sentiment] = 0

    #iterate through each sentence/sentiment pair in the training data
    for sentence, sentiment in sentencesTrain.iteritems():
        if sentence == '':
            continue

        wordList = re.findall(r"[\w']+", sentence) # get word list

        unigramList = wordList
        bigramList = makeNgram(wordList, 2)
        trigramList = makeNgram(wordList, 3)

        for word in (bigramList + unigramList + trigramList): # now calculate over bigrams
            wordTotals['all'] += 1
            if not dictionary.has_key(word):
                dictionary[word] = 1
            wordTotals[sentiment] += 1
            if not freq[sentiment].has_key(word):
                freq[sentiment][word] = 1
            else:
                freq[sentiment][word] += 1


    for word in dictionary:
        #do some smoothing so that minimum count of a word is 1
        for sentiment in sentiments():
            if not freq[sentiment].has_key(word):
                freq[sentiment][word] = 1
            else:
                freq[sentiment][word] += 1

        freqWordAll = 0
        for sentiment in sentiments():
            freqWordAll += freq[sentiment][word]

        # calculate p(word|sentiment)
        for sentiment in sentiments():
            pWord[sentiment][word] = freq[sentiment][word] / float(wordTotals[sentiment])
        pWord['all'][word] = freqWordAll / float(wordTotals['all']) #p(word)


#---------------------------End Training ----------------------------------

#implement naive bayes algorithm
#INPUTS:
#  sentences is a dictonary with sentences associated with sentiment
#  dataName is a string (used only for printing output)
#  pWordPos is dictionary storing p(word|positive) for each word
#     i.e., pWordPos["apple"] will return a real value for p("apple"|positive)
#  pWordNeg is dictionary storing p(word|negative) for each word
#  pWord is dictionary storing p(word)
#  pPos is a real number containing the fraction of positive reviews in the dataset
def testBayes(sentences, dataName, pWord):

    total = {}
    total['all'] = 0
    for sentiment in sentiments():
        total[sentiment] = 0

    correct = {}
    correct['all'] = 0
    for sentiment in sentiments():
        correct[sentiment] = 0

    #for each sentence, sentiment pair in the dataset
    for sentence, sentiment in sentences.iteritems():
        if sentence == '':
            continue

        wordList = re.findall(r"[\w']+", sentence)#collect all words

        unigramList = wordList
        bigramList = makeNgram(wordList, 2)
        trigramList = makeNgram(wordList, 3)

        p = {}
        for s in sentiments():
            p[s] = 0.16

        for word in (trigramList + bigramList + unigramList):
            if pWord['all'].has_key(word):
                for s in sentiments():
                    if pWord[s][word] > 0.00000001:
                        p[s] *= pWord[s][word] * 10000

        total['all'] += 1

        totalProb = float(sum(p.itervalues()))
        threshold = 0.166
        prob = 0
        prob = p[sentiment] / totalProb
        total[sentiment] += 1
        if prob >= threshold:
            correct['all'] += 1
            correct[sentiment] += 1

    print dataName + " Accuracy"

    acc = {}
    acc['all'] = correct['all'] / float(total['all'])
    for sentiment in sentiments():
        acc[sentiment] = correct[sentiment] /  float(total[sentiment])

    print " (ALL)=%0.2f" % acc['all'] + " (%d" % correct['all'] + "/%d" % total['all'] + ")"
    for sentiment in sentiments():
        print " (" + sentiment + ")=%0.2f" % acc[sentiment] + " (%d" % correct[sentiment] + "/%d" % total[sentiment] + ")"


#---------- Main Script --------------------------


sentencesTrain={}
sentencesTest={}

#initialise datasets and dictionaries
readFiles(sentencesTrain,sentencesTest)

pWord={ 'all': {} } # p(W)
for sentiment in sentiments():
    pWord[sentiment] = {} # p(W|sentiment)



#build conditional probabilities using training data
trainBayes(sentencesTrain, pWord)

#run naive bayes classifier on datasets
print "Naive Bayes"
testBayes(sentencesTrain, "Train", pWord)

print

testBayes(sentencesTest,  "Test", pWord)
