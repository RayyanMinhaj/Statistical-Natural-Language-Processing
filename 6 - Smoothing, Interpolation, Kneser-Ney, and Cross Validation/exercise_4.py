from collections import Counter, defaultdict
import string
from itertools import islice

def preprocess_text(text, unk_threshold=1):
    """Preprocesses raw text into tokens suitable for language modeling.
    
    Performs the following steps:
    1. Removes all punctuation and newline characters
    2. Converts text to lowercase
    3. Splits text into word tokens
    4. Replaces rare words (appearing fewer than unk_threshold times) with <UNK>
    
    Args:
        text (str): Raw input text to process
        unk_threshold (int, optional): Minimum count threshold for word retention.
            Words with counts below this will be replaced with <UNK>. Defaults to 1.
    
    Returns:
        list[str]: List of processed tokens with <UNK> replacements

    """
    no_punct = ''.join(char for char in text if char not in string.punctuation)
    no_newlines = no_punct.replace('\n', ' ').replace('\r', ' ')
    lowercase_text = no_newlines.lower()
    tokens = lowercase_text.split()

    word_counts = Counter(tokens)
    all_words=word_counts.most_common(len(tokens))

    words_with_lt_threshold=[]

    for word, count in all_words:
        if count<unk_threshold:
            words_with_lt_threshold.append(word)

    for i in range(0, len(tokens)):
        if tokens[i] in words_with_lt_threshold:
            tokens[i] = '<UNK>'
    
    return tokens



class SmoothingCounter:
    """Language model with multiple smoothing techniques for n-gram probability estimation.
    
    Supports three smoothing methods:
    - Good-Turing estimation
    - Kneser-Ney smoothing (with fixed discount)
    - Add-alpha (Laplace) smoothing
    
    Attributes:
        d (float): Discount parameter for Kneser-Ney smoothing
        alpha (float): Smoothing parameter for add-alpha (Laplace) smoothing
        V (int): Vocabulary size (number of unique unigrams)
    """

    def __init__(self, text, alpha=0, d=0.75):
        """Initializes language model with n-gram counts and smoothing parameters.
        
        Args:
            text (list[str]): Preprocessed list of tokens
            alpha (float, optional): Smoothing parameter for add-alpha (Laplace) method. 
                Defaults to 0 (no smoothing).
        """
        self.alpha = alpha
        self.d = d

        #cuonting unigram, bigram, trigram counts
        self.unigrams = Counter(text)
        self.bigrams = Counter(zip(text, islice(text,1,None)))
        self.trigrams = Counter(zip(text, islice(text,1,None), islice(text,2,None) ))

        self.total_unigrams = sum(self.unigrams.values()) #total num of tokens (its num of values, not the counts)
        self.V = len(self.unigrams) #num of unqiue unigrams

        #for kneser-ney continuation counts
        self.continuation_bigram = defaultdict(set) #w_i -> set(w_(i+1))
        self.continuation_trigram = defaultdict(set) #(w_(i-1) , w_i ) -> set(w_(i+1))
        
        for (w1,w2) in self.bigrams:
            self.continuation_bigram[w2].add(w1)

        for (w1,w2,w3) in self.trigrams:
            self.continuation_trigram[(w1,w2)].add(w3)

        #normalization in continuation probabilities
        self.unique_bigrams = len(self.bigrams)
        self.unique_trigrams = len(self.trigrams)



###################################################################################################


    def prob_good_turing_bigram(self, bigram):
        """Computes Good-Turing smoothed probability for a bigram.
        
        Args:
            bigram (tuple[str, str]): Bigram to calculate probability for
        
        Returns:
            float: Smoothed probability P(w2|w1)
        """
        c = self.bigrams[bigram]
        count_of_counts = Counter(self.bigrams.values())
        
        N_c = count_of_counts.get(c, 0)
        N_c_plus_1 = count_of_counts.get(c + 1, 0)
        
        total_bigrams = sum(self.bigrams.values())
        
        # Sum of counts of all bigrams with the same prefix (w1)
        prefix = bigram[0]
        prefix_count = sum(count for (w1, w2), count in self.bigrams.items() if w1 == prefix)
        
        if c == 0:
            # Probability mass for unseen bigrams, approximated by count_of_counts[1]/total_bigrams
            return count_of_counts.get(1, 0) / total_bigrams
        
        if N_c == 0:
            return 0.0
        
        c_star = (c + 1) * (N_c_plus_1 / N_c)
        
        if prefix_count > 0:
            return c_star / prefix_count
        else:
            return 0.0






    def prob_good_turing_trigram(self, trigram):
        """Computes Good-Turing smoothed probability for a trigram.
        
        Args:
            trigram (tuple[str, str, str]): Trigram to calculate probability for
        
        Returns:
            float: Smoothed probability P(w3|w1,w2)
        """
        #c* = (c+1)x(N_c+1) / N_c

        c = self.trigrams[trigram]
        count_of_counts = Counter(self.trigrams.values())

        N_c = count_of_counts.get(c, 0)
        N_c_plus_1 = count_of_counts.get(c + 1, 0)

        total_trigrams = sum(self.trigrams.values()) #incase trigram has count 0

        # Sum of counts of all trigrams with the same prefix (w1, w2)
        prefix = trigram[:2]
        prefix_count = sum(count for tg, count in self.trigrams.items() if tg[:2] == prefix)

        if c==0:
            return count_of_counts.get(1,0)/total_trigrams
        
        if N_c == 0:
            return 0.0
        
        c_star = (c+1)*(N_c_plus_1/N_c)

        if prefix_count>0:
            return c_star/prefix_count
        else:
            return 0.0



###################################################################################################



    def knprob_bigram(self, bigram):
        """Computes Kneser-Ney smoothed probability for a bigram.
        
        Args:
            bigram (tuple[str, str]): Bigram to calculate probability for
            
        Returns:
            float: Smoothed probability P_KN(w3|w2)
        """
        w1, w2 = bigram

        bigram_count = self.bigrams[(w1, w2)]
        unigram_count = self.unigrams[w1]
        
        if unigram_count == 0:
            return self.unigrams[w2] / self.total_unigrams

        distinct_continuations = sum(1 for (x, y) in self.bigrams if x == w1)
        discounted_prob = max(bigram_count - self.d, 0) / unigram_count
        lambda_weight = (self.d / unigram_count) * distinct_continuations

        unique_left_contexts = len(self.continuation_bigram[w2])
        continuation_prob = unique_left_contexts / self.unique_bigrams

        return discounted_prob + lambda_weight * continuation_prob



    def knprob_trigram(self, trigram):
        """Computes Kneser-Ney smoothed probability for a trigram.
        
        Args:
            trigram (tuple[str, str, str]): Trigram to calculate probability for
            
        Returns:
            float: Smoothed probability P_KN(w3|w1,w2)
        """
        w1, w2, w3 = trigram

        trigram_count = self.trigrams[trigram]
        bigram_count = self.bigrams[(w1,w2)]

        if bigram_count==0:
            return self.knprob_bigram((w2,w3))
        
        distinct_continuations = sum(1 for (x, y, z) in self.trigrams if x == w1 and y == w2)
    
        discounted_prob = max(trigram_count - self.d, 0) / bigram_count
        
        lambda_weight = (self.d / bigram_count) * distinct_continuations

        backoff_prob = self.knprob_bigram((w2, w3))

        P = discounted_prob + lambda_weight * backoff_prob

        return P



###################################################################################################


    def prob_alpha_bigram(self, bigram):
        """Computes add-alpha (Laplace) smoothed bigram probability.
        
        Args:
            bigram (tuple[str, str]): Bigram to calculate probability for
            
        Returns:
            float: Smoothed probability P_alpha(w2|w1)

        """

        w1, w2 = bigram
        count_bigram = self.bigrams[bigram]
        count_unigram = self.unigrams[w1]
        
        P = (count_bigram + self.alpha) / (count_unigram + self.alpha * self.V)
        return P




    def prob_alpha_trigram(self, trigram):
        """Computes add-alpha (Laplace) smoothed trigram probability.
        
        Args:
            trigram (tuple[str, str, str]): Trigram to calculate probability for
            
        Returns:
            float: Smoothed probability P_alpha(w3|w1,w2)
        """
        #P(w3|w2,w1) = C(w1 w2 w3) + alpha / C(w1 w2) + alpha*V 
        
        w1, w2, w3 = trigram
        count_trigram = self.trigrams[trigram]
        count_bigram = self.bigrams[(w1,w2)]

        P = (count_trigram + self.alpha)/(count_bigram + (self.alpha*self.V))

        return P
