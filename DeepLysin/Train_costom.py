from Feature import all_feature,readFasta
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression as LR
from xgboost.sklearn import XGBClassifier as XGBoost
from sklearn.ensemble import ExtraTreesClassifier as ERT
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier as ANN
from pathlib import Path
import time,os
import argparse
Randon_seed = 100


def base_clf(clf,X_train,y_train,model_name,path,n_folds=10):
    ntrain = X_train.shape[0]
    nclass = len(np.unique(y_train))
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Randon_seed)
    base_train = np.zeros((ntrain,nclass))

    for train_index, test_index in kf.split(X_train,y_train):
        kf_X_train,kf_y_train = X_train[train_index],y_train[train_index]
        kf_X_test = X_train[test_index]

        clf.fit(kf_X_train, kf_y_train)
        base_train[test_index] = clf.predict_proba(kf_X_test)
    clf.fit(X_train,y_train)
    joblib.dump(clf, path + f'/base/{model_name}')
    return base_train[:,-1]
    

def process_train(fastafile, pos_num, neg_num, path):
    seqs = readFasta(fastafile)
    y_true = np.array([1 if i<int(pos_num) else 0 for i in range(int(pos_num)+int(neg_num))],dtype=int)
    train_features,feature_index = all_feature(seqs, path)
    print(y_true)
    base_feature = []
    for idx,(k,v) in zip(feature_index,clf_feature_order.items()):
        features = train_features[:,idx]
        for j in v:
            model = eval(j)
            base_proba = base_clf(model,features,y_true,f'{k}_{j[:-4]}.m',path)
            base_feature.append(base_proba)
    return np.array(base_feature).T,y_true


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="Usage Tip;",
                                     description = "Classifier training")
    parser.add_argument("--file", "-f", required = True,
                        help = "input file(.fasta)")
    parser.add_argument("--pos_num", "-p", required=True, help="positive sample number")
    parser.add_argument("--neg_num", "-n", required=True, help="negative sample number")
    parser.add_argument("--model_path", "-m", required=True, help="Model path")
    parser.add_argument('--model_list', nargs='+', help='<Required> Base model', required=True)
    parser.add_argument('--feature_list', nargs='+', help='<Required> Base feature', required=True)
    Args = parser.parse_args()
    
    start_time = time.time()
    njob = 8
    Path(os.path.abspath(Args.model_path) +'/base/').mkdir(exist_ok=True,parents=True)
    Path(os.path.abspath(Args.model_path) +'/Features/').mkdir(exist_ok=True,parents=True)
    Path(os.path.abspath(Args.model_path) +'/Features/' + 'AAEindex/').mkdir(exist_ok=True,parents=True)
    
    with open(os.path.abspath(Args.model_path) +'/Features/AAEindex/AAindex_1.txt','w') as w1:
      line = 'AccNo	A	R	N	D	C	Q	E	G	H	I	L	K	M	F	P	S	T	W	Y	V' + '\n'
      line = line + 'EISD840101	0.25	-1.76	-0.64	-0.72	0.04	-0.69	-0.62	0.16	-0.40	0.73	0.53	-1.10	0.26	0.61	-0.07	-0.26	-0.18	0.37	0.02	0.54' + '\n'
      line = line + 'HOPT810101	-0.5	3.0	0.2	3.0	-1.0	0.2	3.0	0.0	-0.5	-1.8	-1.8	3.0	-1.3	-2.5	0.0	0.3	-0.4	-3.4	-2.3	-1.5' + '\n'
      line = line + 'CHAM810101	0.52	0.68	0.76	0.76	0.62	0.68	0.68	0.00	0.70	1.02	0.98	0.68	0.78	0.70	0.36	0.53	0.50	0.70	0.70	0.76' + '\n'
      line = line + 'EISD860101	0.67	-2.1	-0.6	-1.2	0.38	-0.22	-0.76	0.	0.64	1.9	1.9	-0.57	2.4	2.3	1.2	0.01	0.52	2.6	1.6	1.5' + '\n'
      line = line + 'KYTJ820101	1.8	-4.5	-3.5	-3.5	2.5	-3.5	-3.5	-0.4	-3.2	4.5	3.8	-3.9	1.9	2.8	-1.6	-0.8	-0.7	-0.9	-1.3	4.2' + '\n'
      line = line + 'MITS020101	0	2.45	0	0	0	1.25	1.27	0	1.45	0	0	3.67	0	0	0	0	0	6.93	5.06	0' + '\n'
      line = line + 'DAWD720101	2.5	7.5	5.0	2.5	3.0	6.0	5.0	0.5	6.0	5.5	5.5	7.0	6.0	6.5	5.5	3.0	5.0	7.0	7.0	5.0' + '\n'
      line = line + 'GRAR740102	8.1	10.5	11.6	13.0	5.5	10.5	12.3	9.0	10.4	5.2	4.9	11.3	5.7	5.2	8.0	9.2	8.6	5.4	6.2	5.9' + '\n'
      line = line + 'BIGC670101	52.6	109.1	75.7	68.4	68.3	89.7	84.7	36.3	91.9	102.0	102.0	105.1	97.7	113.9	73.6	54.9	71.2	135.4	116.2	85.1'
      w1.write(line)
    w1.close()
    
    with open(os.path.abspath(Args.model_path) +'/Features/AAEindex/AAindex_2.txt','w') as w2:
      line = 'AccNo	A	R	N	D	C	Q	E	G	H	I	L	K	M	F	P	S	T	W	Y	V' + '\n'
      line = line + 'FAUJ880109	0.	4.	2.	1.	0.	2.	1.	0.	1.	0.	0.	2.	0.	0.	0.	1.	1.	1.	1.	0.' + '\n'
      line = line + 'KLEP840101	0.	1.	0.	-1.	0.	0.	-1.	0.	0.	0.	0.	1.	0.	0.	0.	0.	0.	0.	0.	0.' + '\n'
      line = line + 'FASG760101	89.09	174.20	132.12	133.10	121.15	146.15	147.13	75.07	155.16	131.17	131.17	146.19	149.21	165.19	115.13	105.09	119.12	204.24	181.19	117.15' + '\n'
      w2.write(line)
    w2.close()
    
    
    ERT_clf = ERT(n_estimators=100, random_state = Randon_seed, n_jobs=njob)
    LR_clf = LR(solver='liblinear',random_state=Randon_seed)
    ANN_clf = ANN(max_iter=5000,random_state=Randon_seed)
    KNN_clf = KNN(n_jobs=njob)
    XGB_clf = XGBoost(n_jobs=njob,random_state=Randon_seed)

    base_model = Args.model_list
    base_feature = Args.feature_list
    
    
    clf_feature_order = {}
    
    for i in base_feature:
      clf_feature_order[i] = base_model
    
    print(clf_feature_order)
    meta_features,y = process_train(Args.file, Args.pos_num, Args.neg_num, os.path.abspath(Args.model_path))
    df = pd.DataFrame(meta_features)
    df.to_csv(os.path.abspath(Args.model_path) + '/Features/Base_features.csv',index=False)
    stop_time = time.time()
    print(f'time:{stop_time-start_time}s')
