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
    train_features,feature_index = all_feature(seqs)
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
    Args = parser.parse_args()
    
    start_time = time.time()
    njob = 8
    Path(os.path.abspath(Args.model_path) +'/base/').mkdir(exist_ok=True,parents=True)
    
    ERT_clf = ERT(n_estimators=100, random_state = Randon_seed, n_jobs=njob)
    LR_clf = LR(solver='liblinear',random_state=Randon_seed)
    ANN_clf = ANN(max_iter=5000,random_state=Randon_seed)
    KNN_clf = KNN(n_jobs=njob)
    XGB_clf = XGBoost(n_jobs=njob,random_state=Randon_seed)

    clf_feature_order = {
        "AAE" : ["ERT_clf"],
        "AAI" : ["XGB_clf"],
        "BPNC" : ["ANN_clf"],
        "CTD" : ["LR_clf"],
        "DPC" : ["ANN_clf","KNN_clf"],
        "GTPC" : ["ANN_clf","XGB_clf"]
    }

    meta_features,y = process_train(Args.file, Args.pos_num, Args.neg_num, os.path.abspath(Args.model_path))
    df = pd.DataFrame(meta_features)
    Path(os.path.abspath(Args.model_path) +'/Features/').mkdir(exist_ok=True,parents=True)
    df.to_csv(os.path.abspath(Args.model_path) + '/Features/Base_features.csv',index=False)
    stop_time = time.time()
    print(f'time:{stop_time-start_time}s')
