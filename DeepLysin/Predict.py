import argparse
from pathlib import Path
from Train import Randon_seed
from Feature import all_feature,readFasta
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression as LR
import time,os


def get_base_proba(test_features,feature_index):
    base_feature = []
    for idx,(k,v) in zip(feature_index,clf_feature_order.items()):
        features = test_features[:,idx]
        for j in v:
            model = eval(j)
            base_proba = model.predict_proba(features)[:,-1]
            base_feature.append(base_proba)
    return np.array(base_feature).T

def meta_model(X,y,path):
    meta_clf = LR(solver='liblinear', random_state=Randon_seed)
    meta_clf.fit(X,y)
    joblib.dump(meta_clf,path)

def meta_pred(fastafile, path_meta, path):
    seqs = readFasta(fastafile)
    test_full_features, feature_index = all_feature(seqs, path)
    base_feature = get_base_proba(test_full_features,feature_index)
    meta_clf = joblib.load(path_meta)
    result = meta_clf.predict_proba(base_feature)
    return result

def input_args():
    """
    Usage:
    python PredNeuroP.py -f test.fasta -o ./data/Features/Data1.csv
    """

    parser = argparse.ArgumentParser(usage="Usage Tip;",
                                     description = "Prediction")
    parser.add_argument("--file", "-f", required = True,
                        help = "input test file(.fasta)")
    parser.add_argument("--out", "-o", required=True, help="Output path and filename")
    parser.add_argument("--pos_train_num", "-pr", required=False, type=int, default=2100, help="Train positive sample number")
    parser.add_argument("--neg_train_num", "-nr", required=False, type=int, default=2051, help="Train negative sample number")
    parser.add_argument("--pos_test_num", "-pe", required=False, help="Test positive sample number")
    parser.add_argument("--neg_test_num", "-ne", required=False, help="Test negative sample number")
    parser.add_argument("--model_path", "-m", required=True, help="Model path")
    return parser.parse_args()

if __name__ == '__main__':
    args = input_args()
    Path(os.path.abspath(args.model_path) + '/meta/').mkdir(exist_ok=True,parents=True)
    start_time = time.time()
    
    clf_feature_order = {
        "AAC" : ["ERT_clf","ANN_clf","XGB_clf"],
        "BPNC" : ["KNN_clf","ANN_clf","XGB_clf"],
        "CTD" : ["ANN_clf","XGB_clf"]
    }

    AAC_ERT = joblib.load(os.path.abspath(args.model_path) + '/base/AAC_ERT.m')
    AAC_ANN = joblib.load(os.path.abspath(args.model_path) + '/base/AAC_ANN.m')
    AAC_XGB = joblib.load(os.path.abspath(args.model_path) + '/base/AAC_XGB.m')
    BPNC_KNN = joblib.load(os.path.abspath(args.model_path) + '/base/BPNC_KNN.m')
    BPNC_ANN = joblib.load(os.path.abspath(args.model_path) + '/base/BPNC_ANN.m')
    BPNC_XGB = joblib.load(os.path.abspath(args.model_path) + '/base/BPNC_XGB.m')
    CTD_ANN = joblib.load(os.path.abspath(args.model_path) + '/base/CTD_ANN.m')
    CTD_XGB = joblib.load(os.path.abspath(args.model_path) + '/base/CTD_XGB.m')
    
    if Path(os.path.abspath(args.model_path) + '/meta/Meta.m').exists() is False:
        X_train = pd.read_csv(os.path.abspath(args.model_path) + '/Features/Base_features.csv')
        y = np.array([1 if i<(int(args.pos_train_num)) else 0 for i in range(int(args.pos_train_num)+int(args.neg_train_num))],dtype=int)
        meta_model(X_train, y, os.path.abspath(args.model_path) + '/meta/Meta.m')
    print('**********  Start  **********')
    test_result = meta_pred(args.file, os.path.abspath(args.model_path) + '/meta/Meta.m', os.path.abspath(args.model_path))[:,-1]
    np.savetxt(args.out,test_result,fmt='%.4f',delimiter=',')
    stoptime = time.time()
    print('********** Finished **********')
    print(f'Result file saved in {args.out}')
    print(f'time cost:{stoptime-start_time}s')
    
    if args.pos_test_num and args.neg_test_num !=None:
        from Metric import scores
        y = np.array([1 if i < (int(args.pos_test_num)) else 0 for i in range(int(args.pos_test_num)+int(args.neg_test_num))], dtype=int)
        metr1,metr2 = scores(y,test_result)
        print(metr1)
        print(metr2)
