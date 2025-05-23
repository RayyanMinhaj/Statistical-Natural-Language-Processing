import random
import math

class MillersModel:
    """Implements the Miller's Model"""

    _char_to_prob = {
        " ": 0.2,
        "e": 0.08929,
        "a": 0.06798,
        "r": 0.06065,
        "i": 0.06036,
        "o": 0.05731,
        "t": 0.05561,
        "n": 0.05323,
        "s": 0.04588,
        "l": 0.04391,
        "c": 0.03631,
        "u": 0.02905,
        "d": 0.02707,
        "p": 0.02534,
        "m": 0.0241,
        "h": 0.02402,
        "g": 0.01976,
        "b": 0.01658,
        "f": 0.0145,
        "y": 0.01422,
        "w": 0.01032,
        "k": 0.00882,
        "v": 0.00806,
        "x": 0.00232,
        "z": 0.00218,
        "j": 0.00158,
        "q": 0.00157,
    }

    _last_char_space = False

    def sample(self):
        """Sample a character based on the probability"""
        char = random.choices(
            list(self._char_to_prob.keys()), weights=list(self._char_to_prob.values())
        )[0]

        if not char and self._last_char_space:
            # if we previously sampled a space character,
            # then we sample again until we get a non-space character
            return self.sample()

        self._last_char_space = char == " "

        return char

    def get_prob(self, char):
        """
        Get the probability of a character

        params
            char: The character to get the probability of
        """
        return self._char_to_prob.get(str.lower(char), 0)

    def compute_perplexity(self, text):
        """
        Compute the perplexity of a text

        params
            text: The text to compute the perplexity of
        """

        N=len(text)
        total_prob = 0
        text = text.lower()

        for char in text:
            prob = self.get_prob(char)

            total_prob+=math.log2(prob)
        
        perplexity = 2**((-1/N)*(total_prob))

        return round(perplexity, 4)
