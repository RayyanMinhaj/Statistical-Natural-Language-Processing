from collections import Counter
import numpy as np
from typing import List, Tuple
import math #for log2 func

class NGramLM:
    def __init__(self, train_tokens: List[str], N: int, alpha: float):
        self.N = N
        self.alpha = alpha
        self.train_counts = [Counter(self.get_n_grams(
            train_tokens, n)) for n in range(1, N+1)]
        self.V = set(self.train_counts[0].keys())




    def get_n_grams(self, tokens: List[str], n: int) -> List[Tuple[str]]:
        """
        Generates n-grams from the provided token list.
        """
        tokens_circular = tokens + tokens[:n-1]  # circular n-grams
        n_grams = []
        for i in range(0, len(tokens_circular)-n+1):
            n_gram = tuple(tokens_circular[i:i+n])
            n_grams.append(n_gram)
        return n_grams




    def get_n_gram_count(self, n_gram: Tuple[str]) -> int:
        """
        Returns the count of a specific n-gram from the training data.
        """
        return self.train_counts[self.N-1][n_gram]




    def get_history_count(self, history: Tuple[str]) -> int:
        """
        Returns the count of the history (the first N-1 words) from the training data.
        """
        if len(history) == 0:
            return sum(self.train_counts[0].values())
        return self.train_counts[self.N-2][history]





    def get_smoothed_prob(self, n_gram: Tuple[str]) -> float:
        """
        Returns the smoothed probability for the given n-gram.
        """
        history = n_gram[:self.N-1]
        return (self.get_n_gram_count(n_gram) + self.alpha)/(self.get_history_count(history) + self.alpha*len(self.V))





    def perplexity(self, test_tokens: List[str]):
        """
        Args:
            test_tokens: list of tokens from the test split

        Returns:
            perplexity of test set

        Hint for Implementation: (utilize the functions already implemented for you)
        1. Calculates relative frequencies of test n-grams
        2. Checks if relative frequencies sum to 1 (within tolerance)
        3. Calculates Perplexity i.e. 2 raised to the power of cross-entropy
        """
        #PP = 2^(-1/N * SUM[log2(p(w))])

        ngrams = self.get_n_grams(test_tokens, self.N) #get the trigrams
         
        ngram_counts = Counter(ngrams) #count relative frequency and make sure it sums up to 1
        total_ngrams = sum(ngram_counts.values())
        rel_freqs = {}
        for ngram, count in ngram_counts.items():
            rel_freqs[ngram] = count / total_ngrams

        #check if relative frequency sums up to 1
        total_prob = sum(rel_freqs.values())
        if not np.isclose(total_prob, 1.0, atol=1e-6):
            raise ValueError(f"Relative frequencies sum to {total_prob}, not 1.")
        
        #else calculate perplexity
        cumulative_log_probs = 0
        for gram in ngrams:
            prob = self.get_smoothed_prob(gram)
            cumulative_log_probs += -math.log2(prob)

        cross_entropy = cumulative_log_probs / len(ngrams)
        PP = 2 ** cross_entropy

        return PP

        
