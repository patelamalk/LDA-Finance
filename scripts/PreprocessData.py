"""
**********************************************
Authors:                                     *
    Anand M Goel   : goelanand@gmail.com     *
    Amal Patel     : patelamal.01@gmail.com  *
**********************************************
"""

import os
import argparse
from multiprocessing import cpu_count, Pool
import sys
import time
import pandas as pd
import pickle
import gc
import copy
import re
# Book
from tika import parser as pr
# NLTK stopwords
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
# Spacy lemmatization
import spacy
nlp = spacy.load("en")
nlp.max_length = 10_000_000
# Gensim
import gensim
from gensim.utils import simple_preprocess
# Regex pattern
pattern = re.compile("(\S*@\S*\s?) | (\') | (\s+)")
bigram_mod = None
trigram_mod = None


def read_file(file_path):
    """
    Compiles the regex and removes emails,special characters, new line
    Read and clean data
    """
    f = open(file_path)
    text = str(f.read())
    f.close()
    text = text.replace("\n", " ")
    text = pattern.sub(" ", text)
    return (file_path, text)


def create_df(path):
    """
    Parallelize it to utilize all cores on the machine
    """
    all_paths = [os.path.join(path, i) for i in os.listdir(path)]
    s = time.time()
    pool = Pool(cpu_count())
    res = pool.map(read_file, all_paths)
    pool.close()
    df = pd.DataFrame(res, columns = ["filepath", "text"])
    e = time.time()
    print("Processing {} files : {} mins".format(len(all_paths), (e - s)/60 ))
    return df


def gensim_tok(text):
    """
    Gensim Tokenization
    """
    return gensim.utils.simple_preprocess(str(text), deacc=True)


def remove_stopwords(doc):
    """
    Remove stopwords
    """
    return [word for word in doc if word not in stop_words]


def make_bigrams(doc):
    """
    Make bigrams
    """
    global bigram_mod
    return bigram_mod[doc]


def make_trigrams(texts):
    """
    Make trigrams
    """
    global trigram_mod
    return trigram_mod[bigram_mod[doc]]


def lemmatize(sent, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """
    Lemmatization
    """
    global nlp
    doc = nlp(" ".join(sent))
    return [token.lemma_ for token in doc if token.pos_ in allowed_postags]


def preprocess_data(data_dir, lemmatized_dir, book_path):
  
    # Get memory details of the machine
    mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    mem_gib = mem_bytes/(1024.**3)
    print("\n")
    print("RAM : {} GB".format(mem_gib))
    print("CPU : {} \n".format(cpu_count()))
    
    # Len of the dataset
    print("{} text files in dataset\n".format(len(os.listdir(data_dir))))
    
    # Create dataframe of the data
    df = create_df(data_dir)
    print("DataFrame Head: {}\n".format(df.head()))
    gc.collect()
    
    # Delete df and convert to list
    data = df.text.to_list()
    del df

    # Tokenizing
    t0 = time.time()
    pool = Pool(cpu_count() - 1)
    data_words = pool.map(gensim_tok, data)
    pool.close()
    print("Tokenizing {} docs : {} mins".format(len(data), (time.time()-t0)/60))
    print("Len of Tokenized datawords : {} \n".format(len(data_words)))
    
    # Creating bigram-trigram models
    # Takes fair amount of time
    gc.collect()
    t0 = time.time()
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
    # Faster way to get a sentence clubbed as a trigram/bigram
    global bigram_mod
    global trigram_mod
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    print("Creating bigram-trigram models : {} mins \n".format((time.time() - t0)/60))
    
    # Removing stopwords
    gc.collect()
    t0 = time.time()
    pool = Pool(cpu_count() - 1)
    data_words_nostops = pool.map(remove_stopwords, data_words)
    pool.close()
    del data_words
    print("Removing stopwords {} docs : {} mins".format(len(data), (time.time()-t0)/60))
    print("Len of no stop words list : {} \n".format(len(data_words_nostops)))
    
    # Form bigrams
    gc.collect()
    t0 = time.time()
    pool = Pool(cpu_count() - 1)
    data_words_bigrams = pool.map(make_bigrams, data_words_nostops)
    pool.close()
    del data_words_nostops
    print("Forming bigrams {} docs : {} mins".format(len(data), (time.time()-t0)/60))
    print("Len of bigrams list : {} \n".format(len(data_words_bigrams)))
    
    # Lemmatization
    gc.collect()
    #nlp = spacy.load('en', disable=["parser", "ner"])
    #nlp.max_length = 10000000
    t0 = time.time()
    pool = Pool(cpu_count() - 1)
    data_lemmatized = pool.map(lemmatize, data_words_bigrams)
    pool.close()
    print("Lemmatizing {} docs : {} mins".format(len(data), (time.time() - t0)/60))
    print("Len of lemmatized list : {} \n".format(len(data_lemmatized)))
    
    # Picklize lemmatized data
    # Takes fair amount of time
    t0 = time.time()
    with open(os.path.join(lemmatized_dir, "data_lemmatized.pickle"), "wb") as f:
        pickle.dump(data_lemmatized, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(lemmatized_dir, "./data.pickle"), "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print("Saving data & lemmatized data : {} mins \n".format((time.time() - t0)/60))
    
    # Lemmatize book
    bookstuff = pr.from_file(book_path)
    bookdata = bookstuff['content']
    bookdata = re.sub('\S*@\S*\s?', '', bookdata)
    bookdata = re.sub('\s+', ' ', bookdata)
    bookdata = re.sub("\'", "", bookdata)
    
    bookwords = gensim.utils.simple_preprocess(str(bookdata), deacc=True)
    bookwords_nostops = [word for word in simple_preprocess(str(bookwords)) if word not in stop_words]
    bookwords_bigrams = bigram_mod[bookwords_nostops]
    del bookdata
    del bookwords
    del bookwords_nostops
    
    book_bigrams1 = bookwords_bigrams[0:66254]
    book_bigrams2 = bookwords_bigrams[66254:]
    global nlp
    book_lemmatized1 = [token.lemma_ for token in nlp(" ".join(book_bigrams1)) if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']]
    book_lemmatized2 = [token.lemma_ for token in nlp(" ".join(book_bigrams2)) if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']]
    
    book_lemmatized = book_lemmatized1 + book_lemmatized2
    del book_bigrams1
    del book_bigrams2
    del book_lemmatized1
    del book_lemmatized2
    
    with open(os.path.join(lemmatized_dir, "book_lemmatized.pickle"), "wb") as f:
        pickle.dump(book_lemmatized, f, pickle.HIGHEST_PROTOCOL)
    print("Saved lemmatized book : {}".format(os.path.join(lemmatized_dir, "book_lemmatized.pickle")))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="path/to/data/dir")
    parser.add_argument("--lemmatized_data_dir", type=str, help="/path/to/lemmatized/data/dir")
    parser.add_argument("--book_path", type=str, help="/path/to/pdf")
    args = parser.parse_args()
    
    if not args.book_path:
        sys.exit("Exiting.\nPass --book_path dir")
    book_path = args.book_path
    
    if not args.data_dir:
        sys.exit("Exiting. \npass --data_dir")
    data_dir = args.data_dir
    
    if not args.lemmatized_data_dir:
        sys.exit("Exiting. \npass --lemmatized_dir")
    lemmatized_dir = args.lemmatized_data_dir
    
    if not os.path.exists(data_dir):
        sys.exit("Exiting. \n--datadir doesnot exist")
    
    if not os.path.exists(lemmatized_dir):
        sys.exit("Exiting. \n--lemmatized_data_dir doesnot exist")
    
    preprocess_data(data_dir, lemmatized_dir, book_path)