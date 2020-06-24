cd ../

python3 -m nltk.downloader stopwords
python3 -m nltk.downloader punkt
python3 -m spacy download en

python3 ./utils/mallet_installation.py
tar -xvzf mallet.tar.gz
rm mallet.tar.gz
