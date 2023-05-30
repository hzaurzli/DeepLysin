# DeepLysin
DeepLysin,

# Pipeline
![DeepLysin](https://github.com/hzaurzli/DeepLysin/assets/47686371/97353455-fce7-4e1e-bdcf-52bb4bb26b0d)

# Usage
### 1. Training
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

### 2. Testing
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
python3 PredNeuroP.py -f ./datasets/test_lysin.fa -o data.csv -pr 2100 -nr 2051 -pe 900 -ne 879 -m ./Models/
```
