"""
**********************************************
Authors:                                     *
    Anand M Goel   : goelanand@gmail.com     *
    Amal Patel     : patelamal.01@gmail.com  *
**********************************************
"""

import os
import gc
import sys
import argparse
import pickle
import pandas as pd
import time
from pprint import pprint
import copy
from operator import itemgetter
import matplotlib.pyplot as plt
# Gensim
import gensim
from gensim.models.ldamulticore import LdaMulticore
import gensim.corpora as corpora
from math import log10


def my_enum(c):
    for i, data in enumerate(c):
        yield data

        
def kl_div(x, y):
    s = 0
    for i in range(len(x)):
        s += ( x[i][1] * log10(x[i][1]/y[i][1]) )
    return s
        
    
def doc_topics(lda_multicore, corpus):
    temp = []
    for data in my_enum(corpus):
        row = lda_multicore[data]
        row.sort(key=itemgetter(1), reverse=True)
        topic_num = row[0][0]
        percent_contribution = row[0][1]
        wp = lda_multicore.show_topic(topic_num)
        keywords = ",".join([word for word, prop in wp])
        temp.append([topic_num, percent_contribution, keywords])
    return temp


def analyize_divergence(model_path, lemmatized_dir, csv_path):
    Nwordsfordiv = 1000
    maxtopics = 20
    # Loading lemmatized data
    t0 = time.time()
    data_lemmatized = pickle.load(open(os.path.join(lemmatized_dir, "data_lemmatized.pickle"), "rb"))
    data = pickle.load(open(os.path.join(lemmatized_dir, "data.pickle"), "rb"))
    print("\n")
    print("Loading files : {} mins\n".format((time.time() - t0)/60))
    
    # Creating dictionary
    t0 = time.time()
    id2word = corpora.Dictionary(data_lemmatized)
    print("Creating dictionary : {} mins\n".format((time.time() - t0) / 60))
    
    # Creating corpus
    texts = data_lemmatized
    
    # Term-document frequency
    t0 = time.time()
    corpus = [id2word.doc2bow(text) for text in texts]
    print("Term document frequency : {} mins\n".format((time.time() - t0) / 60))
    
    # Load the LDA model
    model_name = os.path.split(model_path)[-1]
    lda_multicore = gensim.models.ldamulticore.LdaMulticore.load(model_path)
    print("Loaded model : {}\n".format(model_path))
    
    # Dominant topics
    t0 = time.time()
    res = doc_topics(lda_multicore, corpus)
    print(f"Dominant topic/document : {(time.time() - t0)/60} mins\n")
    doc_topics_df = pd.DataFrame(res)
    doc_topics_df.columns = ["Dominant_Topic", "Precent_Contribution", "Keywords"]
    del res
    gc.collect()
    print("Dominant topics : \n{}\n".format(doc_topics_df))
    
    # Group 5 sentences under each topic
    print("Group 5 sentences under each topic:")
    texts = pd.Series(texts)
    doc_topics_df =  pd.concat([doc_topics_df, texts], axis=1)
    doc_topics_df.columns = ['Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    doc_topics_df.index.names = ["Document No"]
    print(doc_topics_df, "\n")
    
    # Most representative document for each topic
    print("Finding most representative document for each topic:")
    topic_doc_df = pd.DataFrame()
    temp = doc_topics_df.groupby('Dominant_Topic')
    for i, grp in temp:
        topic_doc_df = pd.concat([topic_doc_df, grp.sort_values(['Topic_Perc_Contrib'], ascending=[0]).head(1)], axis=0)
    topic_doc_df.reset_index(drop=True, inplace=True)
    topic_doc_df.columns = ["Topic_Num", "Topic_Percent_Contrib", "Keywrds", "Text"]
    print(topic_doc_df, "\n")
    
    # Topic distribution across documents
    # Number of documents for each topic
    topic_counts = doc_topics_df['Dominant_Topic'].value_counts()
    # Percent of documents for each topic
    topic_contribution = round(topic_counts/topic_counts.sum(), 4)
    plt.bar(height=topic_contribution, x=topic_contribution.index)
    plt.xlabel("Topic Number")
    plt.ylabel("Porbabilities")
    plt.title("Topic Distribution")
    plt.savefig(os.path.join(csv_path, model_name + "_Topic Distribution.png"))
    print("Saved Topic Distribution \n")
    
    # Topic Number & Keywords
    print("Topic numbers and Keywords:")
    topic_number_keywords = doc_topics_df[['Dominant_Topic', 'Keywords']]
    print(topic_number_keywords)
    print("\n")
    
    # Saving topic probabilitites
    del doc_topics_df["Text"]
    pprint(doc_topics_df.head())
    doc_topics_df.to_csv(os.path.join(csv_path, model_name + "_Topic_probabilities.csv"), index=False)
    print("Saved Topic distribution : {}\n".format(os.path.join(csv_path, model_name + "_Topic_probabilities.csv")))
    
    # Create word distributions
    model_topics = lda_multicore.show_topics(formatted=False, num_topics=maxtopics, num_words=min(Nwordsfordiv, len(lda_multicore.id2word)))
    topic_df = pd.DataFrame()
    topics_bow = []
    for i, topic_row in enumerate(model_topics):
        topic = topic_row[1]
        topic_bow = []
        for word_tuple in topic:
            prob = word_tuple[1]
            word = word_tuple[0]
            topic_df = topic_df.append(pd.Series([i, word, float(prob)]), ignore_index=True)
            topic_bow.append((lda_multicore.id2word.doc2bow([word])[0][0], float(prob)))
        topics_bow.append(topic_bow)

    topic_df.columns = ["Topic Number", "Word", "Probability"]
    topic_df.to_csv(os.path.join(csv_path, model_name + "_Word_Distribution.csv"))
    print("Saved word distribution : {}\n".format(csv_path, model_name + "_Word_Distribution.csv"))
    
    # Load lemmatized book
    book_lemmatized = pickle.load(open(os.path.join(lemmatized_dir, 'book_lemmatized.pickle'), 'rb')) 
    book_bow = id2word.doc2bow(book_lemmatized)
    
    # Form probabilities for book data
    cbook_bow = []
    norm = sum(frequency for _, frequency in book_bow)
    for wordtuple in book_bow:
        cbook_bow.append((wordtuple[0], wordtuple[1]/norm))
        
    res = []
    for i, topic in enumerate(topics_bow):

        topic_temp = copy.deepcopy(topic)
        book_temp = copy.deepcopy(cbook_bow)

        t_id = list(set(i[0] for i in topic))
        b_id = list(set(i[0] for i in cbook_bow))

        for k in list(set(t_id) - set(b_id)):
            book_temp.append((k, 0.000001))
        book_temp = sorted(book_temp, key=itemgetter(0))

        for j in list(set(b_id) - set(t_id)):
            topic_temp.append((j, 0.000001))
        topic_temp = sorted(topic_temp, key=itemgetter(0))

        print("Topic ", i, " has topic, book divergence : ", kl_div(topic_temp, book_temp))
        print("Topic ", i, " has book, topic divergence : ", kl_div(book_temp, topic_temp))
        print()
        res.append([kl_div(topic_temp, book_temp), 
                    kl_div(book_temp, topic_temp),
                    kl_div(book_temp, topic_temp) + kl_div(topic_temp, book_temp)])
    
    kldiv_df = pd.DataFrame(res, columns=["KLdiv_Topic_Book", "KLdiv_Book_Topic", "Entropy"])
    kldiv_df.index.name = "Topic No"
    kldiv_df.to_csv(os.path.join(csv_path, model_name+"_KL_Divergence.csv"))
    
    print("KL Divergence dataframe : ")
    print(kldiv_df, "\n")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="/path/to/model/")
    parser.add_argument("--lemmatized_data_path", type=str, help="/path/to/lemmatized/data/dir")
    parser.add_argument("--prob_file_path", type=str, help="/path/to/save/probability/csv/")
    args = parser.parse_args()
    
    if not args.prob_file_path:
        sys.exit("Exiting. \nPass --prob_file_path")
    csv_path = args.prob_file_path
    
    if not args.model_path:
        sys.exit("Exiting. \nPass arg --model_path")
    model_path = args.model_path
    
    if not args.lemmatized_data_path:
        sys.exit("Exiting. \nPass arg --lemmatized_data_path") 
    lemmatized_dir = args.lemmatized_data_path
    
    analyize_divergence(model_path, lemmatized_dir, csv_path)