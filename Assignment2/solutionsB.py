import sys
import nltk
import math
import time

import numpy as np

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []

    for sentence in brown_train:
      tokens = [START_SYMBOL + "/" + START_SYMBOL] + [START_SYMBOL + "/" + START_SYMBOL] + sentence.split() + [STOP_SYMBOL + "/" + STOP_SYMBOL]
      words = []
      tags = []
      for token in tokens:
          words += [nltk.tag.util.str2tuple(token)[0]]
          tags += [nltk.tag.util.str2tuple(token)[1]]
      brown_words += [words]
      brown_tags += [tags]

    return brown_words, brown_tags


# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}

    tokens = [item for sublist in brown_tags for item in sublist]
    trigrams = list(nltk.trigrams(tokens))
    condition_pairs = (((w0, w1), w2) for w0, w1, w2 in trigrams)
    fd = nltk.ConditionalFreqDist(condition_pairs)
    pd = nltk.ConditionalProbDist(fd, nltk.MLEProbDist)
    q_values = { item : pd[(item[0],item[1])].logprob(item[2]) for item in trigrams }

    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    tokens = [item for sublist in brown_words for item in sublist]
    fd = nltk.FreqDist(tokens)
    known_words = [ item for item in fd if fd[item] > 5 ]
    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []

    for sentence in brown_words:
        new_sentence = []
        for word in sentence:
            if word in known_words:
                new_sentence += [word]
            else:
                new_sentence += [RARE_SYMBOL]
        brown_words_rare += [new_sentence]

    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    words = [item for sublist in brown_words_rare for item in sublist]
    tags = [item for sublist in brown_tags for item in sublist]

    tagged_words = [(tag,word) for tag,word in zip(tags,words)]

    fd = nltk.ConditionalFreqDist(tagged_words)
    pd = nltk.ConditionalProbDist(fd, nltk.MLEProbDist)

    e_values = { (item[1], item[0]) : pd[item[0]].logprob(item[1]) for item in tagged_words }
    taglist = set(tags)

    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist_1, known_words, q_values, e_values):
    prob = { (0,"*","*") : 0 }
    bp = {}
    taglist = taglist_1 - set(["*", "STOP"])
    sentence = ["*", "*"] + brown_dev_words[1]
    n = len(brown_dev_words[1])

    # k = 1:
    for v in taglist:
      if ("*",v) in e_values:
        prob[(1,"*",v)] = q_values["*","*",v] if ("*","*",v) in q_values else LOG_PROB_OF_ZERO + e_values["*", v]
      else:
        prob[(1,"*",v)] = q_values["*","*",v] if ("*","*",v) in q_values else LOG_PROB_OF_ZERO + e_values[RARE_SYMBOL, v]

    # k = 2:
    for u in taglist:
      for v in taglist:
        e_value = e_values[RARE_SYMBOL, v]
        if ("*",v) in e_values:
          e_value = e_values["*", v]

        q_value= LOG_PROB_OF_ZERO
        if ("*",u,v) in q_values:
          q_value = q_values["*",u,v]

        prob[(2,u,v)] = prob[(1,"*",v)] + q_value + e_value

    # k > 2
    for k in xrange(3, n+1):
      for u in taglist:
        for v in taglist:
          e_value = e_values[RARE_SYMBOL, v]
          if (sentence[k-1],v) in e_values:
            e_value = e_values[sentence[k-1], v]

          prob[(k,u,v)] = max([prob[(k-1,w,u)] + (q_values[w,u,v] if (w,u,v) in q_values else LOG_PROB_OF_ZERO) + e_value for w in taglist])

          print { (k,u,v) : prob[(k-1,w,u)] + (q_values[w,u,v] if (w,u,v) in q_values else LOG_PROB_OF_ZERO) + e_value for w in taglist }
          bp[(k,u,v)] = list(taglist)[np.argmax([prob[(k-1,w,u)] + (q_values[w,u,v] if (w,u,v) in q_values else LOG_PROB_OF_ZERO) + e_value for w in taglist])]

    tags = {}

    taglist_1 = [(u,v) for u in taglist for v in taglist]
    tags[n-1], tags[n] = list(taglist_1)[np.argmax([prob[(n-1,item[0],item[1])] + (q_values[item[0],item[1],"STOP"] if (item[0],item[1],"STOP") in q_values else LOG_PROB_OF_ZERO) for item in taglist_1 ])]

    for k in range(n-2,0,-1):
      tags[k] = bp[k+2,tags[k+1],tags[k+2]]

    # tagged = []
    # return tagged
    return prob, bp, tags

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a
# terminal newline, not a list of tokens.
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
