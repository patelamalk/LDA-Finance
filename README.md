# LDA_Finance
#### Text based analysis of Financial Innovation.
##### Platform: 
``` bash 
Debian 10
```

##### Python version: 
``` python
python 3.7.3
```

##### Java installation:
```bash 
sudo apt update
sudo apt install default-jre
java -version 
sudo apt install default-jdk
javac -version
```

##### For large dataset you might encounter java heap space error:
Append the following line to the bashrc file
```bash
java -Xmx{any number without braces}g   # Xmx54g - allocates 54GB of heap space
```

##### For larger files you may encounter spacy memory error:
After loading nlp from spacy, include max_length in the script
```python
nlp.max_length = 10000000              # any number > 1000000
```

##### Dependency installation:
``` bash 
pip3 install numpy 
pip3 install pandas 
pip3 install nltk 
pip3 install gensim 
pip3 install pyLDAvis 
pip3 install spacy 
pip3 install matplotlib 
pip3 install tika 
pip3 install ipdb
```
##### ***** Have the same directory structure as this repo *****
##### Script usage:
Home dir:
```sh
cd LDA_Finance/
```

```bash 
mv utils
bash unpack_data.sh                     # creates data dir and unzips the dataset there 
bash mallet_installation.sh             # downloads mallet and unzips it 
```

Preprocess Data:
```bash 
python3 preprocess_data.py --datadir=/path/to/dir/ --lemmatized_data_dir=/path/to/lemmatized_data --book_path=/path/to/book.pdf
```
Pass the whole path including home, example below:
```python
# python3 preprocess_data.py --data_dir=/home/patelamal_01/LDA_repo/LDA_Finance/data --lemmatized_data_dir=/home/patelamal_01/LDA_repo/LDA_Finance/lemmatized_data --book_path=/home/patelamal_01/LDA_repo/LDA_Finance/Tidd_Innovation.pdf
```

LDA Model:
```sh
python3 LDAMulticoreModel.py --model_name=ModelName --save_model_path=/path/to/save/model/ --lemmatized_data_path=/path/to/processed/lemmatized/data
```
Pass the whole path including home, example below:
```python
# python3 LDAMulticoreModel.py --model_name=LDA_MC_1 --save_model_path=/home/patelamal_01/LDA_repo/LDA_Finance/model/ --lemmatized_data_path=/home/patelamal_01/LDA_repo/LDA_Finance/lemmatized_data/
```

Analysis & Divergence:
```sh
python3 Analysis_Divergence.py --model_path=/path/to/model/modelname --lemmatized_data_path=/path/to/lemmatized/data/ --prob_file_path=/path/to/save/prob/csv/
```
Pass the whole path including home, example below:
```python
# python3 Analysis_Divergence.py --model_path=/home/patelamal_01/LDA_repo/LDA_Finance/model/LDA_MC_1/LDA_MC_1 --lemmatized_data_path=/home/patelamal_01/LDA_repo/LDA_Finance/lemmatized_data/ --prob_file_path=/home/patelamal_01/LDA_repo/LDA_Finance/probability/
```

##### Authors:
[Dr. Anand Goel](http://www.anandgoel.org/)
[Amal Patel](https://www.linkedin.com/in/patelamalk/)

##### Useful links:
[Gensim Tutorials](https://radimrehurek.com/gensim/auto_examples/index.html)
[Topic Modeling with Gensim](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#1introduction)
[Topic Modeling with mallet](https://programminghistorian.org/en/lessons/topic-modeling-and-mallet)
