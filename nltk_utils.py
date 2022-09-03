import nltk
import numpy as np
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer  = PorterStemmer()



def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_word(tokenzied_sentence, all_words):
    ""
    tokenzied_sentence = [stem(w) for w in tokenzied_sentence]
    bag = np.zeros(len(all_words),dtype=np.float32)
    for idx, w in enumerate (all_words):
        if w in tokenzied_sentence:
            bag[idx]=1.0
    return bag


# a ="How are you?"
# a1= tokenize(a)
# print(a1)

# words = ["organize","orgnizing"]


# for w in words:
#     print(stem(w))

# sentence = ["hello","how","are","you"]
# words = ["hi","hello","I","bye"]
# bog = bag_of_word(sentence,words)
# print(bog)