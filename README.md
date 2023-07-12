# DeepLysin
DeepLysin, easy and fast digging lysin from phages and prophages 

# Pipeline
![DeepLysin](https://github.com/hzaurzli/DeepLysin/assets/47686371/37053d80-1075-42a8-a2cb-b465badb403a)


## Step 1: Lysin_finder: Find putative lysins
### Install the software
```
# install env
conda env create -f lysin_finder_env.yml

# activate env
source activate lysin_finder_env
```

If your perl version is not 5.22, please install perl=5.22
```
# install perl 5.22
conda install -c anaconda perl=5.22
```

### Find putative lysins
***Notice:*** **Genome fasta file suffix is ```.fna```, to see example in Data fold**

***If Genome fasta file suffix is not ```.fna```, you can run :***
```
# Rename
python rename_suffix.py -p ./data/ -s fasta

usage: rename_suffix.py [-h] -p PATH -s SUFFIX
suffix rename
options:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  genome sequence path
  -s SUFFIX, --suffix SUFFIX
                        old suffix/To be modified suffix

```
For example, if fasta file's suffix is '.fasta', run ```python rename_suffix.py -p ./data/ -s fasta```, that can change suffix '.fasta' into '.fna'

And then, run ```lysin_finder.py```:
```
# activate env
source activate lysin_finder_env

# run
python lysin_finder.py
  -p /.../input_path/                               # genome sequence path, genome fasta file suffix is '.fna'
  -t Bacteria                                       # prokka kingdom type    
  -hd ./db/hmm/lysin_reported.hmm                   # hmmer database path
  -rl ./db/hmm/lysin_reported.txt                   # reported lysin structures(hmm files)
  -cd ./db/cazy/db/                                 # cazy database path
  -wkdir ./test/                                    # work directory
  -ml 10000                                         # lower proteins molecular weight
  -mu 40000                                         # upper proteins molecular weight
  -hde ./db/hmm/lysin_reported_EAD.hmm              # EAD hmmer database path
  -rle ./db/hmm/lysin_reported_EAD.txt              # reported lysin EAD structures(hmm files)
  -hdp ./db/hmm/lysin_reported_peptidases.hmm       # peptidases hmmer database path
  -rlp ./db/hmm/lysin_reported_peptidases.txt       # reported lysin peptidases structures(hmm files)
  -bp B                                             # 'B' for bacteria, 'P' for phage
```

## Step 2: DeepLysin bactericidal activity scoring

### Usage
#### A.Basic usage
##### 1. Training
**Step 1: random sample**
```
usage: Sample_selection.py [-h] -a ALL_FA [-tr TRAIN] [-te TEST]
                           [-p PART_SAMPLE] -num NUMBER -s SEED
Sample fasta file
optional arguments:
  -h, --help            show this help message and exit
  -a ALL_FA, --all_fa ALL_FA
                        All the fasta dataset for your study
  -tr TRAIN, --train TRAIN
                        Training dataset
  -te TEST, --test TEST
                        Testing dataset
  -p PART_SAMPLE, --part_sample PART_SAMPLE
                        Part of all datasets that you can sample randomly
  -num NUMBER, --number NUMBER
                        Dataset size(number) for training dataset,remain for
                        testing dataset.Or for part dataset
  -s SEED, --seed SEED  Random seed
  
# Example for devide training set and testing set
## For positive sample
python3 Sample_selection.py -a ./datasets/pos_lysin.fa -tr ./datasets/pos_train_lysin.fa -te ./datasets/pos_test_lysin.fa -n 2100 -s 12345
## For negative sample
python3 Sample_selection.py -a ./datasets/neg_lysin.fa -tr ./datasets/neg_train_lysin.fa -te ./datasets/neg_test_lysin.fa -n 2051 -s 12345
## Make training set and testing set
cd datasets
cat pos_train_lysin.fa neg_train_lysin.fa > train_lysin.fa
cat pos_test_lysin.fa neg_test_lysin.fa > test_lysin.fa


# Example to make part dataset
python3 Sample_selection.py -a ./datasets/pos_lysin.fa -p ./datasets/part.fa -n 500 -s 12345
```

**Step 2: training model**
```
usage: Usage Tip;
Classifier training
optional arguments:
  -h, --help            show this help message and exit
  --file FILE, -f FILE  input file(.fasta)
  --pos_num POS_NUM, -p POS_NUM
                        positive sample number
  --neg_num NEG_NUM, -n NEG_NUM
                        negative sample number
  --model_path MODEL_PATH, -m MODEL_PATH
                        Model path


# Example
## Make model fold to save model
mkdir Models
## Training
python3 Train.py -f ./datasets/train_lysin.fa -p 2100 -n 2051 -m ./Models/ 
```

##### 2. Testing
```
usage: Usage Tip;
Prediction
optional arguments:
  -h, --help            show this help message and exit
  --file FILE, -f FILE  input test file(.fasta)
  --out OUT, -o OUT     Output path and filename
  --pos_train_num POS_TRAIN_NUM, -pr POS_TRAIN_NUM
                        Train positive sample number
  --neg_train_num NEG_TRAIN_NUM, -nr NEG_TRAIN_NUM
                        Train negative sample number
  --pos_test_num POS_TEST_NUM, -pe POS_TEST_NUM
                        Test positive sample number
  --neg_test_num NEG_TEST_NUM, -ne NEG_TEST_NUM
                        Test negative sample number
  --model_path MODEL_PATH, -m MODEL_PATH
                        Model path


# Example
python3 Predict.py -f ./datasets/test_lysin.fa -o data.csv -pr 2100 -nr 2051 -pe 900 -ne 879 -m ./Models/
```

##### 3. Prediction
```
# Example
python3 Predict.py -f ./datasets/target_lysin.fa -o data.csv -m ./Models/
```

#### B.Custom usage
##### 1. Training
**Step 1: random sample**
```
usage: Sample_selection.py [-h] -a ALL_FA [-tr TRAIN] [-te TEST]
                           [-p PART_SAMPLE] -num NUMBER -s SEED
Sample fasta file
optional arguments:
  -h, --help            show this help message and exit
  -a ALL_FA, --all_fa ALL_FA
                        All the fasta dataset for your study
  -tr TRAIN, --train TRAIN
                        Training dataset
  -te TEST, --test TEST
                        Testing dataset
  -p PART_SAMPLE, --part_sample PART_SAMPLE
                        Part of all datasets that you can sample randomly
  -num NUMBER, --number NUMBER
                        Dataset size(number) for training dataset,remain for
                        testing dataset.Or for part dataset
  -s SEED, --seed SEED  Random seed
  
# Example for devide training set and testing set
## For positive sample
python3 Sample_selection.py -a ./datasets/pos_lysin.fa -tr ./datasets/pos_train_lysin.fa -te ./datasets/pos_test_lysin.fa -n 2100 -s 12345
## For negative sample
python3 Sample_selection.py -a ./datasets/neg_lysin.fa -tr ./datasets/neg_train_lysin.fa -te ./datasets/neg_test_lysin.fa -n 2051 -s 12345
## Make training set and testing set
cd datasets
cat pos_train_lysin.fa neg_train_lysin.fa > train_lysin.fa
cat pos_test_lysin.fa neg_test_lysin.fa > test_lysin.fa


# Example to make part dataset
python3 Sample_selection.py -a ./datasets/pos_lysin.fa -p ./datasets/part.fa -n 500 -s 12345
```

**Step 2: training model**
```
usage: Usage Tip;
Classifier training
optional arguments:
  -h, --help            show this help message and exit
  --file FILE, -f FILE  input file(.fasta)
  --pos_num POS_NUM, -p POS_NUM
                        positive sample number
  --neg_num NEG_NUM, -n NEG_NUM
                        negative sample number
  --model_path MODEL_PATH, -m MODEL_PATH
                        Model path
  --model_list MODEL_LIST [MODEL_LIST ...]
                        <Required> Base model
  --feature_list FEATURE_LIST [FEATURE_LIST ...]
                        <Required> Base feature



# Example
## Make model fold to save model
mkdir Models
## Training
python3 Train_costom.py -f ./datasets/train_lysin.fa -p 2100 -n 2051 -m ./Model/ --model_list ANN_clf XGB_clf --feature_list AAE AAI
```
##### Model type
| Model    |     Param     |
|----------|:-------------:|
| Extremely randomized trees |  ERT_clf |
| logistic regression |    LR_clf   |
| Artificial neural network | ANN_clf |
| Extreme gradient boosting | XGB_clf |
| K-nearest neighbor | KNN_clf |

##### Feature type
| Feature    |     Param     |
|----------|:-------------:|
| Amino acid index |  AAI |
| In grouped amino acid composition |  GAAC  |
| Composition–transition–distribution | CTD |
| Amino acid entropy | AAE |
| Amino acid composition | AAC |
| Binary Profile-Based feature | BPNC |


##### 2. Testing
```
usage: Usage Tip;
Prediction
optional arguments:
  -h, --help            show this help message and exit
  --file FILE, -f FILE  input test file(.fasta)
  --out OUT, -o OUT     Output path and filename
  --pos_train_num POS_TRAIN_NUM, -pr POS_TRAIN_NUM
                        Train positive sample number
  --neg_train_num NEG_TRAIN_NUM, -nr NEG_TRAIN_NUM
                        Train negative sample number
  --pos_test_num POS_TEST_NUM, -pe POS_TEST_NUM
                        Test positive sample number
  --neg_test_num NEG_TEST_NUM, -ne NEG_TEST_NUM
                        Test negative sample number
  --model_path MODEL_PATH, -m MODEL_PATH
                        Model path
  --feature_model FEATURE_MODEL [FEATURE_MODEL ...]
                        <Required> Base feature model


# Example
python3 Predict_costom.py -f ./datasets/test_lysin.fa -o data.csv -pr 2100 -nr 2051 -pe 900 -ne 879 -m ./Models/ --feature_model AAE_ERT AAI_XGB BPNC_ANN CTD_LR DPC_ANN DPC_KNN GTPC_ANN GTPC_XGB
```
```--feature_model``` ***is the combination of models and features, please see model type and feature type***

##### 3. Prediction
```
# Example
python3 Predict_costom.py -f ./datasets/target_lysin.fa -o data.csv -m ./Models/ --feature_model AAE_ERT AAI_XGB BPNC_ANN CTD_LR DPC_ANN DPC_KNN GTPC_ANN GTPC_XGB
```
```--feature_model``` ***is the combination of models and features, please see model type and feature type***


# Reference database download
Baidu：
  Links：https://pan.baidu.com/s/1coUbBGpiSHmxgy418XWQDw
  Password：smrz

# Cition
If this software is useful, please cite [https://github.com/hzaurzli/DeepLysin](https://github.com/hzaurzli/DeepLysin)
