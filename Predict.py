import argparse
from pathlib import Path
from Base_classifier import Randon_seed
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

def meta_model(X,y):
    meta_clf = LR(solver='liblinear', random_state=Randon_seed)
    meta_clf.fit(X,y)
    joblib.dump(meta_clf,'./Models/meta/Meta.m')

def meta_pred(fastafile):
    seqs = readFasta(fastafile)
    test_full_features, feature_index = all_feature(seqs)
    base_feature = get_base_proba(test_full_features,feature_index)
    meta_clf = joblib.load('./Models/meta/Meta.m')
    result = meta_clf.predict_proba(base_feature)
    return result

def input_args():
    """
    Usage:
    python PredNeuroP.py -f test.fasta -o ./data/Features/Data1.csv
    """

    parser = argparse.ArgumentParser(usage="Usage Tip;",
                                     description = "PreNeuroP Predict")
    parser.add_argument("--file", "-f", required = True,
                        help = "input file(.fasta)")
    parser.add_argument("--out", "-o", required=True, help="output path and filename")
    parser.add_argument("--pos_num", "-p", required=True, help="test positive sample number")
    parser.add_argument("--neg_num", "-n", required=True, help="test negative sample number")
    parser.add_argument("--model_path", "-m", required=True, help="Model path")
    return parser.parse_args()

if __name__ == '__main__':
    args = input_args()
    Path(os.path.abspath(args.model_path) + '/meta/').mkdir(exist_ok=True,parents=True)
    start_time = time.time()
    clf_feature_order = {
      "AAE" : ["AAE_ERT"],
      "AAI" : ["AAI_XGB"],
      "BPNC" : ["BPNC_ANN"],
      "CTD" : ["CTD_LR"],
      "DPC" : ["DPC_ANN","DPC_KNN"],
      "GTPC" : ["GTPC_ANN","GTPC_XGB"]
    }
    
    AAE_ERT = joblib.load(os.path.abspath(args.model_path) + '/base/AAE_ERT.m')
    AAI_XGB = joblib.load(os.path.abspath(args.model_path) + '/base/AAI_XGB.m')
    BPNC_ANN = joblib.load(os.path.abspath(args.model_path) + '/base/BPNC_ANN.m')
    CTD_LR = joblib.load(os.path.abspath(args.model_path) + '/base/CTD_LR.m')
    DPC_ANN = joblib.load(os.path.abspath(args.model_path) + '/base/DPC_ANN.m')
    DPC_KNN = joblib.load(os.path.abspath(args.model_path) + '/base/DPC_KNN.m')
    GTPC_ANN = joblib.load(os.path.abspath(args.model_path) + '/base/GTPC_ANN.m')
    GTPC_XGB = joblib.load(os.path.abspath(args.model_path) + '/base/GTPC_XGB.m')
    
    if Path(os.path.abspath(args.model_path) + '/meta/Meta.m').exists() is False:
        X_train = pd.read_csv(os.path.abspath(args.model_path) + '/Features/Base_features.csv')
        y = np.array([1 if i<(int(args.pos_num)) else 0 for i in range(int(args.pos_num)+int(args.neg_num))],dtype=int)
        meta_model(X_train,y)
    print('**********  Start  **********')
    test_result = meta_pred(args.file)[:,-1]
    np.savetxt(args.out,test_result,fmt='%.4f',delimiter=',')
    stoptime = time.time()
    print('********** Finished **********')
    print(f'Result file saved in {args.out}')
    print(f'time cost:{stoptime-start_time}s')

    from Metric import scores
    y = np.array([1 if i < (int(args.pos_num)) else 0 for i in range(int(args.pos_num)+int(args.neg_num))], dtype=int)
    metr1,metr2 = scores(y,test_result)
    print(metr1)
    print(metr2)
