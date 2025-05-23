# TODO: Add your necessary imports here
from typing import Union
import nltk
from nltk.corpus import brown
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

def load_corpus():
    """Load `Brown` corpus from NLTK"""
    nltk.download('brown')

    #for sake of simplicity well be using the news category as the list of tokens
    text = []
    text = brown.words(categories='news')
    #print(text)

    lower_text = []
    for word in text:
        lower_text.append(word.lower())

    #print(lower_text)

    return lower_text


def get_bigram_freqs(text: list):
    bigram = {}

    for i in range(0, len(text)-1):
        bigram_set = (text[i], text[i+1])
        bigram[bigram_set] = bigram.get(bigram_set, 0) + 1
    
    return bigram


def get_top_n_probabilities(text: list, context_words: Union[str, list], n: int):
    """ Get top `n` following words to `context_words` and their probabilities

    Args:
    text -- A list of tokens to be used as the corpus text
    context_words -- A `str` containing the context word(s) to be considered
    n    -- An `int` that indicates how many tokens to evaluate
    """

    bigrams = get_bigram_freqs(text)

    #we basically need count("to", any_word)/count("to") for first 50 words
    
    #FIRST WE DO FOR "to"
    #P(wi,wi-1) = COUNT("to", "any_word")/COUNT("to")
    to_dict_probs = {}
    to_count = 0
    for word in text:
        if word == context_words[1]:
            to_count+=1

    for key,value in bigrams.items():
        if key[0] == context_words[1]:
            to_dict_probs[f'P("{key[1]}" | "{context_words[1]}")'] = round(value/to_count, 6)

    to_dict_probs = dict(sorted(to_dict_probs.items(), key=lambda item: item[1], reverse=True))
    to_dict_probs = dict(list(to_dict_probs.items())[:n])


    #NOW WE DO FOR "the"
    #P(wi,wi-1) = COUNT("the", "any_word")/COUNT("the")
    the_dict_probs = {}
    the_count = 0
    for word in text:
        if word == context_words[0]:
            the_count+=1
    
    for key,value in bigrams.items():
        if key[0] == context_words[0]:
            the_dict_probs[f'P("{key[1]}" | "{context_words[0]}")'] = round(value/the_count, 6)


    the_dict_probs = dict(sorted(the_dict_probs.items(), key=lambda item: item[1], reverse=True))
    the_dict_probs = dict(list(the_dict_probs.items())[:n])

    #df1 = pd.DataFrame(list(to_dict_probs.items()), columns=["P(Wi | Wi-1 = 'to')", "Probability"])
    #df2 = pd.DataFrame(list(the_dict_probs.items()), columns=["P(Wi | Wi-1 = 'the')", "Probability"])

    #df_combined = pd.concat([df1.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)

    return to_dict_probs, the_dict_probs
    #return df_combined.to_string(index=False)
   


#since we used a dataframe to be able to print prettier, we are changing the signature from dict to a df
def get_entropy(top_n_to: dict, top_n_the:dict):
    """ Get entropy of distribution of top `n` bigrams """
    to_H_x = 0 #entropy of "to"
    the_H_x = 0 #entropy of "the"

    for probability in top_n_to.values():
        to_H_x += -(probability*math.log2(probability)) 


    for probability in top_n_the.values():
        the_H_x += -(probability*math.log2(probability)) 

    
    return to_H_x, the_H_x


def plot_top_n(top_n_to: dict, top_n_the:dict):
    """ Plot top `n` """
    prob_values_to = list(top_n_to.values())
    prob_values_the = list(top_n_the.values())

    #since we dont want the x-axis to say which P(Wi | Wi-1) were plotting so well just use a range (although we already know theyre 50 values)
    plt.figure(figsize=(10,6))
    x = range(len(prob_values_to))
    plt.plot(x, prob_values_to, label='P(Wi | Wi-1 = "to")', marker='o')
    plt.plot(x, prob_values_the, label='P(Wi | Wi-1 = "the")', marker='x')

    plt.legend()

    plt.xlabel('P(Wi | Wi-1)')
    plt.ylabel('Probability')
    plt.title('Probability distribution for the 50 most frequent tokens')

    plt.show()



def get_perplexity(text:str, top_n_probs:dict):
    tokens = []
    tokens = text.split()
    #print("len of tokens(should be 2): ", len(tokens))
    #print(top_n_probs)

    key = f'P("{tokens[1]}" | "{tokens[0]}")'
    prob_value = top_n_probs.get(key)

    if prob_value == None:
        print(f'Perplexity of "{text}" = infinite! (bigram does not exist)')
        return

    else:
        #print(f'this is the probability of P({tokens[1]} | {tokens[0]}) = {prob_value} ')

        perplexity = 2**((-1/len(tokens))*(math.log2(prob_value)))

        print(f'Perplexity of "{text}" = {round(perplexity,4)}')





def get_mean_rank(text:str, top_n_probs:dict):
    #formula for Mean Rank is 1/N x SUM(rank - or wherever it appears)
    tokens = []
    tokens = text.split()
    N = len(tokens)

    key = f'P("{tokens[1]}" | "{tokens[0]}")'
    
    rank = 0

    for i,k in enumerate(top_n_probs.keys()):
        if k == key:
            rank = i+1        

    mean_rank = (1/N)*(rank)
    if mean_rank == 0:
        print(f'Mean Rank of "{text}" = {round(mean_rank,4)} (bigram does not exist)')
    else:
        print(f'Mean Rank of "{text}" = {round(mean_rank,4)}')


