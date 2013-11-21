#!/usr/bin/env python
import re, random, math, collections, itertools

#------------- Function Definitions ---------------------

def readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia):

    #reading pre-labeled movie reviews and splitting into lines

    angrySentences = []
    disgustedSentences = []
    fearfulSentences = []
    happySentences = []
    sadSentences = []
    surprisedSentences = []

    txt = open('emotions/angry.txt', 'r')
    angrySentences = re.split(r'\n', txt.read())

    txt = open('emotions/disgusted.txt', 'r')
    disgustedSentences = re.split(r'\n', txt.read())

    txt = open('emotions/fearful.txt', 'r')
    fearfulSentences = re.split(r'\n', txt.read())

    txt = open('emotions/happy.txt', 'r')
    happySentences = re.split(r'\n', txt.read())

    txt = open('emotions/sad.txt', 'r')
    sadSentences = re.split(r'\n', txt.read())

    txt = open('emotions/surprised.txt', 'r')
    surprisedSentences = re.split(r'\n', txt.read())

    allSentences = angrySentences + disgustedSentences + fearfulSentences + \
                   happySentences + sadSentences + surprisedSentences

    #Create single sentiment dictionary, where words have value 1 if positive and -1 if negative:

    sentimentDictionary={} #initialise dictionary

    for sentences in angrySentences:
        sentimentDictionary[sentence] = "angry"

    for sentences in disgustedSentences:
        sentimentDictionary[sentence] = "disgusted"

    for sentences in fearfulSentences:
        sentimentDictionary[sentence] = "fearful"

    for sentences in happySentences:
        sentimentDictionary[sentence] = "happy"

    for sentences in sadSentences:
        sentimentDictionary[sentence] = "sad"

    for sentences in surprisedSentences:
        sentimentDictionary[sentence] = "surprised"

    #create Training and Test Datsets

    #create 90-10 split of training and test data, with sentiment labels
    sentenceTrain={}
    sentimentTest={}

    for sentence, sentiment in sentimentDictionary.iteritems():
        if random.randint(1,10)<2:
            sentencesTest[sentence] = sentiment
        else:
            sentencesTrain[sentence] = sentiment

#----------------------------End of data initialisation ----------------#

    angry
    disgusted
    fearful
    happy
    sad
    surprised

#calculates p(W|Positive), p(W|Negative) and p(W) for all words in training data
def trainBayes(sentencesTrain, pWordAngry, pWordDisgusted, pWordFearful,
        pWordHappy, pWordSad, pWordSusprised, pWord):

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
        wordList = re.findall(r"[\w']+", sentence) # get word list

        #TO DO:
        #Populate bigramList by concatenating adjacent words in the sentence.
        #You might want to seperate the words by _ for readability, so bigrams such as:
        #You_might, might_want, want_to, to_seperate

        bigramList=[] #initialise bigramList

        for word in bigramList: # now calculate over bigrams
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
        if not freqDisgusted.has_key(word):
            freqDisgusted[word] = 1
        if not freqFearful.has_key(word):
            freqFearful[word] = 1
        if not freqHappy.has_key(word):
            freqHappy[word] = 1
        if not freqSad.has_key(word):
            freqSad[word] = 1
        if not freqSurprised.has_key(word):
            freqSurprised[word] = 1

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

#Print out n most useful predictors
def mostUseful(pWordPos, pWordNeg, pWord, n):
    predictPower={}
    for word in pWord:
        if pWordNeg[word]<0.0000001:
            predictPower=1000000000
        else:
            predictPower[word]=pWordPos[word] / pWordNeg[word]

    sortedPower = sorted(predictPower, key=predictPower.get)
    head, tail = sortedPower[:n], sortedPower[len(predictPower)-n:]
    print "NEGATIVE:"
    print head
    print "\nPOSITIVE:"
    print tail

#implement naive bayes algorithm
#INPUTS:
#  sentencesTest is a dictonary with sentences associated with sentiment
#  dataName is a string (used only for printing output)
#  pWordPos is dictionary storing p(word|positive) for each word
#     i.e., pWordPos["apple"] will return a real value for p("apple"|positive)
#  pWordNeg is dictionary storing p(word|negative) for each word
#  pWord is dictionary storing p(word)
#  pPos is a real number containing the fraction of positive reviews in the dataset
def testBayes(sentencesTest, dataName, pWordPos, pWordNeg, pWord, pPos):
    pNeg=1-pPos

    #These variables will store results (you do not need them)
    total=0
    correct=0
    totalpos=0
    totalneg=0
    correctpos=0
    correctneg=0

    #for each sentence, sentiment pair in the dataset
    for sentence, sentiment in sentencesTest.iteritems():
        wordList = re.findall(r"[\w']+", sentence)#collect all words

        #TO DO: Exactly what you did in the training function:
        #Populate bigramList by concatenating adjacent words in the sentence.

        bigramList=[]


        pPosW=pPos
        pNegW=pNeg
        for word in bigramList:
            if pWord.has_key(word):
                if pWord[word]>0.00000001:
                    #repeated multiplication can make pPosW and pNegW very small
                    #So I multiply them by a large number to keep the arithmatic
                    #sensible. It doesn't change the maths when you
                    #calculate "prob"
                    pPosW *=pWordPos[word]*100000
                    pNegW *=pWordNeg[word]*100000

        prob=pPosW/float(pPosW+pNegW)
        total+=1

        if sentiment=="positive":
            totalpos+=1
            if prob>0.5:
                correct+=1
                correctpos+=1
            else:
                correct+=0
        else:
            totalneg+=1
            if prob<=0.5:
                correct+=1
                correctneg+=1
            else:
                correct+=0

    acc=correct/float(total)
    print dataName + " Accuracy (All)=%0.2f" % acc + " (%d" % correct + "/%d" % total + ")"
    accpos=correctpos/float(totalpos)
    accneg=correctneg/float(totalneg)
    print dataName + " Accuracy (Pos)=%0.2f" % accpos + " (%d" % correctpos + "/%d" % totalpos + ")"
    print dataName + " Accuracy (Neg)=%0.2f" % accneg + " (%d" % correctneg + "/%d" % totalneg + ")\n"



# This is a simple classifier that uses a sentiment dictionary to classify
# a sentence. For each word in the sentence, if the word is in the positive
# dictionary, it adds 1, if it is in the negative dictionary, it subtracts 1.
# If the final score is above a threshold, it classifies as "Positive",
# otherwise as "Negative"
def testDictionary(sentencesTest, dataName, sentimentDictionary, threshold):
    total=0
    correct=0
    totalpos=0
    totalneg=0
    correctpos=0
    correctneg=0
    for sentence, sentiment in sentencesTest.iteritems():
        Words = re.findall(r"[\w']+", sentence)
        score=0
        for word in Words:
            if sentimentDictionary.has_key(word):
               score+=sentimentDictionary[word]

        total+=1
        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
            else:
                correct+=0
        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
            else:
                correct+=0

    acc=correct/float(total)
    print dataName + " Accuracy (All)=%0.2f" % acc + " (%d" % correct + "/%d" % total + ")"
    accpos=correctpos/float(totalpos)
    accneg=correctneg/float(totalneg)
    print dataName + " Accuracy (Pos)=%0.2f" % accpos + " (%d" % correctpos + "/%d" % totalpos + ")"
    print dataName + " Accuracy (Neg)=%0.2f" % accneg + " (%d" % correctneg + "/%d" % totalneg + ")\n"




#---------- Main Script --------------------------


sentimentDictionary={} # {} initialises a dictionary [hash function]
sentencesTrain={}
sentencesTest={}
sentencesNokia={}

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
        pWordHappy, pWordSad, pWordSusprised, pWord)

#run naive bayes classifier on datasets
print "Naive Bayes"
testBayes(sentencesTrain,  "Films (Train Data)\t", pWordPos, pWordNeg, pWord,0.5)
testBayes(sentencesTest,  "Films  (Test Data)\t", pWordPos, pWordNeg, pWord,0.5)
