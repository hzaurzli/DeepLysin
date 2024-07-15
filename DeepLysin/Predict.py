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
    python Predict.py -f test.fasta -o ./data/Features/Data1.csv
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
        "AAC": ["AAC_ERT", "AAC_ANN", "AAC_XGB", "AAC_KNN", "AAC_LR"],
        "BPNC": ["BPNC_ERT", "BPNC_ANN", "BPNC_XGB", "BPNC_KNN", "BPNC_LR"],
        "CTD": ["CTD_ERT", "CTD_ANN", "CTD_XGB", "CTD_KNN", "CTD_LR"],
        "AAE": ["AAE_ERT", "AAE_ANN", "AAE_XGB", "AAE_KNN", "AAE_LR"],
        "AAI": ["AAI_ERT", "AAI_ANN", "AAI_XGB", "AAI_KNN", "AAI_LR"],
        "GAAC": ["GAAC_ERT", "GAAC_ANN", "GAAC_XGB", "GAAC_KNN", "GAAC_LR"],
    }
    
    path = os.path.abspath(args.model_path)

    AAC_ERT = joblib.load(path + '/base/AAC_ERT.m')
    AAC_ANN = joblib.load(path + '/base/AAC_ANN.m')
    AAC_XGB = joblib.load(path + '/base/AAC_XGB.m')
    AAC_KNN = joblib.load(path + '/base/AAC_KNN.m')
    AAC_LR = joblib.load(path + '/base/AAC_LR.m')

    BPNC_ERT = joblib.load(path + '/base/BPNC_ERT.m')
    BPNC_ANN = joblib.load(path + '/base/BPNC_ANN.m')
    BPNC_XGB = joblib.load(path + '/base/BPNC_XGB.m')
    BPNC_KNN = joblib.load(path + '/base/BPNC_KNN.m')
    BPNC_LR = joblib.load(path + '/base/BPNC_LR.m')

    CTD_ERT = joblib.load(path + '/base/CTD_ERT.m')
    CTD_ANN = joblib.load(path + '/base/CTD_ANN.m')
    CTD_XGB = joblib.load(path + '/base/CTD_XGB.m')
    CTD_KNN = joblib.load(path + '/base/CTD_KNN.m')
    CTD_LR = joblib.load(path + '/base/CTD_LR.m')

    AAE_ERT = joblib.load(path + '/base/AAE_ERT.m')
    AAE_ANN = joblib.load(path + '/base/AAE_ANN.m')
    AAE_XGB = joblib.load(path + '/base/AAE_XGB.m')
    AAE_KNN = joblib.load(path + '/base/AAE_KNN.m')
    AAE_LR = joblib.load(path + '/base/AAE_LR.m')

    AAI_ERT = joblib.load(path + '/base/AAI_ERT.m')
    AAI_ANN = joblib.load(path + '/base/AAI_ANN.m')
    AAI_XGB = joblib.load(path + '/base/AAI_XGB.m')
    AAI_KNN = joblib.load(path + '/base/AAI_KNN.m')
    AAI_LR = joblib.load(path + '/base/AAI_LR.m')

    GAAC_ERT = joblib.load(path + '/base/GAAC_ERT.m')
    GAAC_ANN = joblib.load(path + '/base/GAAC_ANN.m')
    GAAC_XGB = joblib.load(path + '/base/GAAC_XGB.m')
    GAAC_KNN = joblib.load(path + '/base/GAAC_KNN.m')
    GAAC_LR = joblib.load(path + '/base/GAAC_LR.m')


    if Path(os.path.abspath(args.model_path) + '/meta/Meta.m').exists() is False:
        X_train = pd.read_csv(os.path.abspath(args.model_path) + '/Features/Base_features.csv')
        y = np.array([1 if i<(int(args.pos_train_num)) else 0 for i in range(int(args.pos_train_num)+int(args.neg_train_num))],dtype=int)
        meta_model(X_train, y, os.path.abspath(args.model_path) + '/meta/Meta.m')
    print('**********  Start  **********')
    test_result = meta_pred(args.file, os.path.abspath(args.model_path) + '/meta/Meta.m', os.path.abspath(args.model_path))[:,-1]
    np.savetxt(os.path.dirname(os.path.abspath(args.out)) + '/tmpLysinActivity.txt',test_result,fmt='%.4f',delimiter=',')
    
    with open(args.file) as fa:
      fa_dict = {}
      for line in fa:
          line = line.replace('\n', '')
          if line.startswith('>'):
              seq_name = line[1:]
              fa_dict[seq_name] = ''
          else:
              fa_dict[seq_name] += line.replace('\n', '')
    fa.close()

    lis = []
    with open(os.path.dirname(os.path.abspath(args.out)) + '/tmpLysinActivity.txt') as ac:
      for i in ac:
          i = i.replace('\n', '')
          lis.append(i)
    ac.close()

    for i_1 in range(0, len(lis)):
      key = list(fa_dict.keys())[i_1]
      val = [fa_dict.get(key, [])] + [lis[i_1]]
      fa_dict[key] = val

    with open(args.out, 'w') as f:
      for key in fa_dict:
          lines = key + '\t' + fa_dict[key][0] + '\t' + fa_dict[key][1] + '\n'
          print(lines)
          f.write(lines)
    f.close()

    os.remove(os.path.dirname(os.path.abspath(args.out)) + '/tmpLysinActivity.txt')
    
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
        from Metric import scores
        y = np.array([1 if i < (int(args.pos_test_num)) else 0 for i in range(int(args.pos_test_num)+int(args.neg_test_num))], dtype=int)
        metr1,metr2 = scores(y,test_result)
        print(metr1)
        print(metr2)
