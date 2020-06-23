# LDA_Finance
Applying LDA to create a metric for financial innovation in financial companies. 
<br />
Folder structure is: <br />
+-- LDA_Finance/ <br />
&nbsp;&nbsp;&nbsp;+-- utils/ <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- unpack_data.sh <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- monitor_memory.sh <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- mallet_installation.sh <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- mallet_installation.py <br />
&nbsp;&nbsp;&nbsp;+-- scripts/ <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- preprocess.py <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- lda_multicore_model.py <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- divergence.py <br />
&nbsp;&nbsp;&nbsp;+-- lemmatized_data/ <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+--  <br />
&nbsp;&nbsp;&nbsp;+-- data/ <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- 1_10K.txt <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- .. <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- .. <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- 1000_10K.txt <br />
&nbsp;&nbsp;&nbsp;+-- Text_Filings.zip <br />
&nbsp;&nbsp;&nbsp;+-- README.md <br />
<br />
<br />
Dataset structure: <br />
Text_Filings.zip <br />
    +-- 1_10K.txt <br />
    +-- .. <br />
    +-- .. <br />
    +-- 1000_10K.txt <br />
<br />
<br />
Platform: Debian 10 / Any Linux Distro  <br />
<br />
<br />
Python version: python 3.7.3  <br />
<br />
<br />
Java installation: <br />
```bash 
sudo apt update
sudo apt install default-jre
java -version 
sudo apt install default-jdk
javac -version
```
<br />     
<br />
For large dataset you might encounter java heap space error: <br />
Append the following line to the bashrc file: <br />
```bash
java -Xmx{any number without braces}g   # Xmx54g - allocates 54GB of heap space
```
<br />
<br /> 
For larger files you may encounter spacy memory error: <br />
after loading nlp from spacy, include in the script <br />
```python
nlp.max_length = 10000000              # any number > 1000000
```
<br />
Dependency installation: <br />
```bash 
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
<br />
<br />
Script usage: LDA_Finance/              # home dir <br />
<br />
********* Execute command in terminal ************ <br />
```bash 
mkdir lemmatized_data                   # Skip this command if the directory exists 
mv utils
bash unpack_data.sh                     # creates data dir and unzips the dataset there 
bash mallet_installation.sh             # downloads mallet and unzips it 
```
************************************************** <br />
<br />
<br /> 
**************** Preprocess ********************** <br />
```bash 
python3 preprocess_data.py --datadir=/path/to/dir/ --lemmatized_data_dir=/path/to/lemmatized_data
```
************************************************** <br />
# Pass the whole path including home, example below <br />
# python3 preprocess_data.py --data_dir=/home/patelamal_01/LDA_repo/LDA_Finance/data -lemmatized_data_dir=/home/patelamal_01/LDA_repo/LDA_Finance/lemmatized_data <br />
<br />
<br />
***************** LDA Model ********************** <br />
```bash
mkdir model                             # Skip this step if the directory exists <br />
<br />
python3 LDAMulticoreModel.py --model_name=ModelName --save_model_path=/path/to/save/model/ --lemmatized_data_path=/path/to/processed/lemmatized/data <br />
```
************************************************** <br />
# Pass the whole path including home, example below <br />
# python3 LDAMulticoreModel.py --model_name=LDA_MC_1 --save_model_path=/home/patelamal_01/LDA_repo/LDA_Finance/model/ --lemmatized_data_path=/home/patelamal_01/LDA_repo/LDA_Finance/lemmatized_data/ <br />
<br />
<br />
********** Analysis and Divergence *************** <br />

************************************************** <br />
