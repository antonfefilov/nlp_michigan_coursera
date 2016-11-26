import math
import nltk
import time

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

def ngrams(n, corpus):
  grams = []

  if n == 1:
    for sentence in corpus:
      tokens = sentence.split()
      grams += list(nltk.ngrams(tokens, n)) + [(STOP_SYMBOL,)]
  else:
    for sentence in corpus:
      tokens = sentence.split() + [STOP_SYMBOL]
      grams += list(nltk.ngrams(tokens, n, pad_left=True, left_pad_symbol=START_SYMBOL))

  return grams

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    print "Calculating unigram prob"
    unigrams = ngrams(1, training_corpus)
    fd = nltk.FreqDist(unigrams)
    pd = nltk.MLEProbDist(fd)
    unigram_p = { item : pd.logprob(item) for item in fd }

    print "Calculating bigram prob"
    bigrams = ngrams(2, training_corpus)
    fd = nltk.ConditionalFreqDist(bigrams)
    pd = nltk.ConditionalProbDist(fd, nltk.MLEProbDist)
    bigram_p = { item : pd[item[0]].logprob(item[1]) for item in bigrams }

    print "Calculating trigram prob"
    trigrams = ngrams(3, training_corpus)
    condition_pairs = (((w0, w1), w2) for w0, w1, w2 in trigrams)
    fd = nltk.ConditionalFreqDist(condition_pairs)
    pd = nltk.ConditionalProbDist(fd, nltk.MLEProbDist)
    trigram_p = { item : pd[(item[0],item[1])].logprob(item[2]) for item in trigrams }

    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc.
def score(ngram_p, n, corpus):
    scores = []

    if n == 1:
        for sentence in corpus:
            prob = 0
            tokens = list(nltk.ngrams(sentence.split(), n)) + [(STOP_SYMBOL,)]
            for token in tokens:
                if token in ngram_p:
                    prob += ngram_p[token]
                else:
                    prob = MINUS_INFINITY_SENTENCE_LOG_PROB
                    break
            scores.append(prob)

        return scores

    else:
        for sentence in corpus:
            prob = 0
            tokens = list(nltk.ngrams(sentence.split() + [STOP_SYMBOL], n, pad_left=True, left_pad_symbol=START_SYMBOL))
            for token in tokens:
                if token in ngram_p:
                    prob += ngram_p[token]
                else:
                    prob = MINUS_INFINITY_SENTENCE_LOG_PROB
                    break
            scores.append(prob)

        return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []

    for sentence in corpus:
        prob = 0
        tokens_tri = list(nltk.ngrams(sentence.split() + [STOP_SYMBOL], 3, pad_left=True, left_pad_symbol=START_SYMBOL))

        for token in tokens_tri:
            temp = 0
            if token in trigrams:
                temp += 0.3333333*(2**trigrams[token])
            if (token[1],token[2]) in bigrams:
                temp += 0.3333333*(2**bigrams[(token[1],token[2])])
            if (token[2],) in unigrams:
                temp += 0.3333333*(2**unigrams[(token[2],)])

            if temp > 0:
                prob += math.log(temp, 2)
            else:
                prob = MINUS_INFINITY_SENTENCE_LOG_PROB
                break
        scores.append(prob)

    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close()

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
