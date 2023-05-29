from Bio import SeqIO
from random import sample
import argparse

def read_data(fpath):
    dna_data = open(fpath).readlines()
    fasta_dic = {}
    id = ''
    seq = ''
    for line in dna_data:
        line = line.strip()
        if len(line)>0:#跳过空行
            if line[0] == '>':
                id = line
                fasta_dic[id] = ''
                seq = ''
            else:
                seq += line
                fasta_dic[id] = seq
    return fasta_dic

def sample_seq(all,some,num,seed):   
    random.seed(int(seed))
    with open(some,'w') as w:
        with open(all) as f:
            seqs = SeqIO.parse(f, "fasta")
            samps = ((seq.name, seq.seq) for seq in sample(list(seqs),int(num)))
            for samp in samps:
                print(">{}\n{}".format(*samp))
                line = ">{}\n{}".format(*samp) + '\n'
                w.write(line)
        f.close()
    w.close()

def main(all_fa,train_fa,test_fa,num):
    all = read_data(all_fa)
    sample_seq(all_fa,train_fa,num)

    train = read_data(train_fa)
    diff = list(all.keys() - train.keys())

    with open(test_fa,'w') as w:
        for key in all:
            if key in diff:
                line = key + '\n' + all[key] + '\n'
                w.write(line)
    w.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sample fasta file")
    parser.add_argument("-a", "--all_fa", required=True, type=str, help="All fasta dataset")
    parser.add_argument("-tr", "--train", required=False, type=str, help="training dataset")
    parser.add_argument("-te", "--test", required=False, type=str, help="testing dataset")
    parser.add_argument("-p", "--part_sample", required=False, type=str, help="testing dataset")
    parser.add_argument("-num", "--number", required=True, type=int, help="Dataset size")
    parser.add_argument("-s", "--seed", required=True, type=int, default=123, help="Random seed")
    Args = parser.parse_args()

    if Args.part_sample != None and Args.train or Args.test == None:
        sample_seq(Args.all_fa,Args.part_sample,int(Args.number))

    else:
        main(Args.all_fa,Args.train,Args.test,int(Args.number))


