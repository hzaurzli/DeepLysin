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
    parser.add_argument('--feature_model', nargs='+', help='<Required> Base feature model', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = input_args()
    Path(os.path.abspath(args.model_path) + '/meta/').mkdir(exist_ok=True,parents=True)
    start_time = time.time()
    
    fmodel = args.feature_model
    dict_clf = {}
    
    for i in fmodel:
      feature = i.split('_')[0]
      model = i.split('_')[1]
      if feature in dict_clf.keys():
        dict_clf[feature].append(feature + '_' + model)
      else:
        dict_clf[feature] = []
        dict_clf[feature].append(feature + '_' + model)
        
    for key in dict_clf:
      dict_tmp = {}
      for item in dict_clf[key]:
          feature = item.split('_')[0]
          model = item.split('_')[1]
          dict_tmp[model] = feature
      dict_2 = dict(sorted(dict_tmp.items(), key=lambda i: i[0]))
  
      lis = []
      for i in dict_2:
          word = dict_2[i] + '_' + i
          lis.append(word)
      dict_clf[dict_2[i]] = lis
      
    clf_feature_order = dict(sorted(dict_clf.items(), key=lambda i: i[0]))
        
    for key in clf_feature_order:
      for item in clf_feature_order[key]:
        exec(item + ' = joblib.load(' + '"' + os.path.abspath(args.model_path) + '/base/' + item + '.m")')
        
    print(clf_feature_order)
    
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
