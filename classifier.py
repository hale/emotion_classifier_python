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

def readFiles(sentimentDictionary,sentencesTrain,sentencesTest):

    #reading pre-labeled movie reviews and splitting into lines

    sentences = {}
    for sentiment in sentiments():
        txt = open('emotions/' + sentiment + '.txt')
        sentences[sentiment] = re.split(r'\n', txt.read())

    for sentiment,sentences in sentences.iteritems():
        for sentence in sentences:
            sentimentDictionary[sentence] = sentiment
            if random.randint(1,10)<2:
                sentencesTest[sentence] = sentiment
            else:
                sentencesTrain[sentence] = sentiment

#----------------------------End of data initialisation ----------------#

#calculates p(W|Positive), p(W|Negative) and p(W) for all words in training data
def trainBayes(sentencesTrain, pWordAngry, pWordDisgusted, pWordFearful,
        pWordHappy, pWordSad, pWordSurprised, pWord):

    angryFeatures = []
    disgustedFeatures = []
    fearfulFeatures = []
    happyFeatures = []
    sadFeatures = []
    surprisedFeatures = []

    freqAngry = {}
    freqDisgusted = {}
    freqFearful = {}
    freqHappy = {}
    freqSad = {}
    freqSurprised = {}

    dictionary = {}

    allWordsTot = 0
    angryWordsTot = 0
    disgustedWordsTot = 0
    fearfulWordsTot = 0
    happyWordsTot = 0
    sadWordsTot = 0
    surprisedWordsTot = 0

    #iterate through each sentence/sentiment pair in the training data
    for sentence, sentiment in sentencesTrain.iteritems():
        if sentence == '':
            continue

        wordList = re.findall(r"[\w']+", sentence) # get word list

        unigramList = wordList
        bigramList = makeNgram(wordList, 2)
        trigramList = makeNgram(wordList, 3)

        for word in (bigramList + unigramList + trigramList): # now calculate over bigrams
            allWordsTot += 1 # keeps count of total words in dataset
            if not dictionary.has_key(word):
                dictionary[word] = 1
            if sentiment=="angry":
                angryWordsTot += 1
                if not freqAngry.has_key(word):
                    freqAngry[word] = 1
                else:
                    freqAngry[word] += 1
            if sentiment=="disgusted":
                disgustedWordsTot += 1
                if not freqDisgusted.has_key(word):
                    freqDisgusted[word] = 1
                else:
                    freqDisgusted[word] += 1
            if sentiment=="fearful":
                fearfulWordsTot += 1
                if not freqFearful.has_key(word):
                    freqFearful[word] = 1
                else:
                    freqFearful[word] += 1
            if sentiment=="happy":
                happyWordsTot += 1
                if not freqHappy.has_key(word):
                    freqHappy[word] = 1
                else:
                    freqHappy[word] += 1
            if sentiment=="sad":
                sadWordsTot += 1
                if not freqSad.has_key(word):
                    freqSad[word] = 1
                else:
                    freqSad[word] += 1
            if sentiment=="surprised":
                surprisedWordsTot += 1
                if not freqSurprised.has_key(word):
                    freqSurprised[word] = 1
                else:
                    freqSurprised[word] += 1


    for word in dictionary:
        #do some smoothing so that minimum count of a word is 1
        if not freqAngry.has_key(word):
            freqAngry[word] = 1
        else:
            freqAngry[word] += 1
        if not freqDisgusted.has_key(word):
            freqDisgusted[word] = 1
        else:
            freqDisgusted[word] += 1
        if not freqFearful.has_key(word):
            freqFearful[word] = 1
        else:
            freqFearful[word] += 1
        if not freqHappy.has_key(word):
            freqHappy[word] = 1
        else:
            freqHappy[word] += 1
        if not freqSad.has_key(word):
            freqSad[word] = 1
        else:
            freqSad[word] += 1
        if not freqSurprised.has_key(word):
            freqSurprised[word] = 1
        else:
            freqSurprised[word] += 1

        # Calculate p(word|angry)
        pWordAngry[word] = freqAngry[word] / float(angryWordsTot)

        # Calculate p(word|disgusted)
        pWordDisgusted[word] = freqDisgusted[word] / float(disgustedWordsTot)

        # Calculate p(word|fearful)
        pWordFearful[word] = freqFearful[word] / float(fearfulWordsTot)

        # Calculate p(word|happy)
        pWordHappy[word] = freqHappy[word] / float(happyWordsTot)

        # Calculate p(word|sad)
        pWordSad[word] = freqSad[word] / float(sadWordsTot)

        # Calculate p(word|surprised)
        pWordSurprised[word] = freqSurprised[word] / float(surprisedWordsTot)

        # Calculate p(word)
        pWord[word] = (freqAngry[word] + freqDisgusted[word] +
        freqFearful[word] + freqHappy[word] + freqSad[word] +
        freqSurprised[word]) / float(allWordsTot)

#---------------------------End Training ----------------------------------

#implement naive bayes algorithm
#INPUTS:
#  sentencesTest is a dictonary with sentences associated with sentiment
#  dataName is a string (used only for printing output)
#  pWordPos is dictionary storing p(word|positive) for each word
#     i.e., pWordPos["apple"] will return a real value for p("apple"|positive)
#  pWordNeg is dictionary storing p(word|negative) for each word
#  pWord is dictionary storing p(word)
#  pPos is a real number containing the fraction of positive reviews in the dataset
def testBayes(sentencesTest, dataName, pWordAngry, pWordDisgusted,
        pWordFearful, pWordHappy, pWordSad, pWordSurprise, pWord):

    #These variables will store results (you do not need them)
    total=0
    correct=0

    totalAngry=0
    totalDisgusted=0
    totalFearful=0
    totalHappy=0
    totalSad=0
    totalSurprised=0

    correctAngry=0
    correctDisgusted=0
    correctFearful=0
    correctHappy=0
    correctSad=0
    correctSurprised=0


    #for each sentence, sentiment pair in the dataset
    for sentence, sentiment in sentencesTest.iteritems():
        if sentence == '':
            continue

        wordList = re.findall(r"[\w']+", sentence)#collect all words

        unigramList = wordList
        bigramList = makeNgram(wordList, 2)
        trigramList = makeNgram(wordList, 3)

        pAngry = pDisgusted = pFearful = pHappy = pSad = pSurprised = 0.16

        for word in (trigramList + bigramList + unigramList):
            if pWord.has_key(word):
                if pWord[word]>0.00000001:
                    pAngry *=pWordAngry[word]*10000
                    pDisgusted *=pWordDisgusted[word]*10000
                    pFearful *=pWordFearful[word]*10000
                    pHappy *=pWordHappy[word]*10000
                    pSad *=pWordSad[word]*10000
                    pSurprised *=pWordSurprised[word]*10000

        total+=1

        totalProb = float(pAngry + pDisgusted + pFearful + pHappy + pSad + pSurprised)
        if totalProb == 0:
          totalProb = 0.00000001
        threshold = 0.166
        prob = 0
        if sentiment=="angry":
            prob=pAngry/totalProb
            totalAngry+=1
            if prob>=threshold:
                correct+=1
                correctAngry+=1
        if sentiment=="disgusted":
            prob=pDisgusted/totalProb
            totalDisgusted+=1
            if prob>=threshold:
                correct+=1
                correctDisgusted+=1
        if sentiment=="fearful":
            prob=pFearful/totalProb
            totalFearful+=1
            if prob>=threshold:
                correct+=1
                correctFearful+=1
        if sentiment=="happy":
            prob=pHappy/totalProb
            totalHappy+=1
            if prob>=threshold:
                correct+=1
                correctHappy+=1
        if sentiment=="sad":
            prob=pSad/totalProb
            totalSad+=1
            if prob>=threshold:
                correct+=1
                correctSad+=1
        if sentiment=="surprised":
            prob=pSurprised/totalProb
            totalSurprised+=1
            if prob>=threshold:
                correct+=1
                correctSurprised+=1


    print dataName + " Accuracy"

    acc=correct/float(total)

    accAngry = correctAngry / float(totalAngry)
    accDisgusted = correctDisgusted / float(totalDisgusted)
    accFearful = correctFearful / float(totalFearful)
    accHappy = correctHappy / float(totalHappy)
    accSad = correctSad / float(totalSad)
    accSurprised = correctSurprised / float(totalSurprised)

    print "  (All)=%0.2f" % acc + " (%d" % correct + "/%d" % total + ")"
    print "  (Angry)=%0.2f" % accAngry + " (%d" % correctAngry + "/%d" % totalAngry + ")"
    print "  (Disgusted)=%0.2f" % accDisgusted + " (%d" % correctDisgusted + "/%d" % totalDisgusted + ")"
    print "  (Fearful)=%0.2f" % accFearful + " (%d" % correctFearful + "/%d" % totalFearful + ")"
    print "  (Happy)=%0.2f" % accHappy + " (%d" % correctHappy + "/%d" % totalHappy + ")"
    print "  (Sad)=%0.2f" % accSad + " (%d" % correctSad + "/%d" % totalSad + ")"
    print "  (Surprised)=%0.2f" % accSurprised + " (%d" % correctSurprised + "/%d" % totalSurprised + ")"

#---------- Main Script --------------------------


sentimentDictionary={} # {} initialises a dictionary [hash function]
sentencesTrain={}
sentencesTest={}

#initialise datasets and dictionaries
readFiles(sentimentDictionary,sentencesTrain,sentencesTest)

pWord={} # p(W)
pWordAngry={} # p(W|Angry)
pWordDisgusted={} # p(W|Disgusted)
pWordFearful={} # p(W|Fearful)
pWordHappy={} # p(W|Happy)
pWordSad={} # p(W|Sad)
pWordSurprised={} # p(W|Surprised)


#build conditional probabilities using training data
trainBayes(sentencesTrain, pWordAngry, pWordDisgusted, pWordFearful,
        pWordHappy, pWordSad, pWordSurprised, pWord)

#run naive bayes classifier on datasets
print "Naive Bayes"
testBayes(sentencesTrain,  "Train", pWordAngry,
        pWordDisgusted, pWordFearful, pWordHappy, pWordSad,
        pWordSurprised, pWord)

print

testBayes(sentencesTest,  "Test", pWordAngry,
        pWordDisgusted, pWordFearful, pWordHappy, pWordSad,
        pWordSurprised, pWord)
