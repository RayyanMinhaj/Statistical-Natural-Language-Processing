from copy import deepcopy
import numpy as np
from typing import List
import math 
from collections import defaultdict, Counter
import nltk
import string
from sklearn.model_selection import train_test_split as sk_train_test_split

nltk.download('treebank')

def load_and_preprocess_data(max_ngram_order=10):

  corpus=list(nltk.corpus.treebank.sents())
  cleaned_corpus=[]
  for sent in corpus:
    #Remove punctuations
    words = [word.translate(str.maketrans(dict.fromkeys(string.punctuation))).lower() for word in sent]
    #Remove empty tokens after cleaning
    words=" ".join(words).split()

    #Removing sentences with less words than our highest ngram order
    #This has to be edited if increasing ngram order higher than 10
    if len(words)>=max_ngram_order:
      cleaned_corpus.append(words)

  return cleaned_corpus




def make_vocab(corpus,top_n):
  '''
  Make the top_n frequent vocabulary from a corpus
  Input: corpus - List[List[str]]
         top_n - int
  Output: Vocabulary - List[str]
  '''
  all_words = []
  for sentence in corpus:
     for word in sentence:
        all_words.append(word)

  word_counts = Counter(all_words)
  
  top_words = []
  for word, count in word_counts.most_common(top_n):
     top_words.append(word)

  return top_words





def restrict_vocab(corpus,vocab):
  '''
  Make the corpus fit inside the vocabulary using <unk>
  Input: corpus - List[List[str]]
         vocab  - List[str]
  Output: Vocabulary_restricted_corpus - List[List[str]]
  '''
  for i in range(0,len(corpus)):
     for j in range(0, len(corpus[i])):
        if corpus[i][j] not in vocab:
           corpus[i][j] = '<unk>'
  
  return corpus

def train_test_split(corpus, split=0.7):
  '''Splits the corpus using a 70:30 ratio. Do not randomize anything here. use the original order
  Input: List[List[str]]
  Output: List[List[str]],List[List[str]]'''
  train, test = sk_train_test_split(corpus, train_size=split, shuffle=False) #keeping shuffle false since data isnt randomized yet.
  return train, test












class Interpolated_Model:
    
    def __init__(self, train_sents: List[List[str]], test_sents: List[List[str]], alpha=0,order=2):
        """ 
        :param train_sents: list of sents from the train section of your corpus
        :param test_sents: list of sents from the test section of your corpus
        :param alpha :  Smoothing factor for laplace smoothing
        :function perplexity() : Calculates perplexity on the test set
        Tips:  Try out the Counter() module from collections to Count ngrams. 
        """
        self.alpha = alpha
        self.order=order
        self.interpolation_weight=1/self.order
        
        #Counting the total number of words in the corpus
        total_words=sum([len(sent) for sent in train_sents])
        self.train_counts=[Counter() for i in range(self.order+1)]

        for sentence in train_sents:
          for ord_ in range(1,self.order+1):
            self.train_counts[ord_]+=Counter(self._get_n_grams(sentence, ord_))

        assert sum(self.train_counts[1].values())==total_words, 'Not all unigrams accounted'

        #At Oth order, return the total number of words for proper normalization
        self.train_counts[0]={():total_words}

        #Getting vocabulary size for laplace smoothing
        self.vocab_size=len(self.train_counts[1])

        #Getting higest order ngrams from the test set
        self.test_ngrams = [self._get_n_grams(sent, self.order) for sent in test_sents]



    def _get_n_grams(self, tokens: List[str], n: int):
      '''
      gets the ngrams out for an arbitrary n value. 
      input: list of tokens
      '''
      n_grams = []
      if n == 0:
          n_grams = [tuple([t]) for t in tokens]
      else:
          for i in range(len(tokens)-n+1):
              n_gram = tuple(tokens[i:i+n])
              n_grams.append(n_gram)
      return n_grams    




    def laplace_prob(self,ngram):
      '''returns the log proabability of an ngram. Adjust this function for Laplace Smoothing'''
      n = len(ngram)

      count_ngram = self.train_counts[n][ngram]
      count_prefix = self.train_counts[n - 1][ngram[:-1]] if n > 1 else self.train_counts[0][()]
      smoothed_prob = (count_ngram + self.alpha) / (count_prefix + self.alpha * self.vocab_size)
      return smoothed_prob
  




    def interpolated_logprob(self,ngram):
      '''
      calculates the interpolated log probability of a given n-gram using the Laplace smoothed probabilities.
      '''
      total_log_prob = 0.0
      for i in range(1, self.order + 1):
          sub_ngram = ngram[-i:]
          prob = self.laplace_prob(sub_ngram)
          total_log_prob += (1 / self.order) * math.log(prob)
      
      return total_log_prob






    def perplexity(self):
      """ returns the perplexity of the language model for n-grams with n=n """
      total_log_prob = 0.0
      total_ngrams = 0

      for sentence_ngrams in self.test_ngrams:
          for ngram in sentence_ngrams:
              log_prob = self.interpolated_logprob(ngram)
              total_log_prob += log_prob
              total_ngrams += 1

      avg_log_prob = total_log_prob / total_ngrams
      perplexity = math.exp(-avg_log_prob)
      
      return perplexity
