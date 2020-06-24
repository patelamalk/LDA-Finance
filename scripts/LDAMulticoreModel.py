"""
**********************************************
Authors:                                     *
    Anand M Goel   : goelanand@gmail.com     *
    Amal Patel     : patelamal.01@gmail.com  *
**********************************************
"""

import os
import pickle
import sys
import argparse
from multiprocessing import cpu_count, Pool
import time
# pyLDAvis
import pyLDAvis
import pyLDAvis.gensim
# Gensim
from gensim.models.ldamulticore import LdaMulticore
import gensim.corpora as corpora
from gensim.models import CoherenceModel
# Define the # of topics
maxtopics = 20



def lda_multicore(lemmatized_dir, save_model_path, model_name):
    """
    Using gensim's inbuilt multicore version
    """
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
    
    # LDA model
    # Took ~150 mins on a 16 CPU, ~100 GB RAM machine
    print("Starting LDA training... \n")
    t0 = time.time()
    lda_multicore = LdaMulticore(corpus=corpus, id2word=id2word, num_topics=maxtopics, workers=cpu_count() - 2, passes=20, iterations=200)
    print("LDA Multicore : {} mins\n".format((time.time() - t0)/60))
    
    # Coherence score
    coherence_multicore = CoherenceModel(model=lda_multicore, texts = data_lemmatized, dictionary=id2word, coherence='u_mass')
    print("Coherence Score : {}\n".format(coherence_multicore.get_coherence()))
    
    # Save the LDA model
    model_path = os.path.join(save_model_path, model_name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    lda_multicore.save(os.path.join(model_path, model_name))
    print("Saved model to dir : {}".format(os.path.join(model_path, model_name)))
    
    # Saving visualization
    vis = pyLDAvis.gensim.prepare(lda_multicore, corpus, id2word)
    pyLDAvis.save_html(vis, os.path.join(model_path, "{}.html".format(model_name)))
    print("Saved visualization to : {}".format(os.path.join(model_path, "{}.html".format(model_name))))
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lemmatized_data_path", type=str, help="/path/to/lemmatized/data/dir")
    parser.add_argument("--save_model_path", type=str, help="/path/to/save/LDAmodel/")
    parser.add_argument("--model_name", type=str, help="Name of the model")
    args = parser.parse_args()
    
    if not args.model_name:
        sys.exit("Exiting. \nPass the model name")
    model_name = args.model_name
    
    if not args.lemmatized_data_path:
        sys.exit("Exiting. \nPass arg --lemmatized_data_path") 
    lemmatized_dir = args.lemmatized_data_path
        
    if not args.save_model_path:
        sys.exit("Exiting. \nPass arg --save_model_path")
    save_model_path = args.save_model_path
    
    if not os.path.exists(lemmatized_dir):
        sys.exit("Exiting.\n--lemmatized_data_path doesnot exist")
        
    if not os.path.exists(save_model_path):
        sys.exit("Exiting.\n--save_model_path doesnot exist")
        
    lda_multicore(lemmatized_dir, save_model_path, model_name)