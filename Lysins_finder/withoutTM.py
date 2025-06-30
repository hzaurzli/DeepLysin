import argparse
from Bio import SeqIO
from Bio import AlignIO
from Bio import pairwise2 as pw2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import os,sys,re,time
import random
import subprocess as sub
from subprocess import *
import subprocess as sub
import glob
import shutil
import biolib
import operator


class tools:
    def __init__(self):
        self.prokka = 'prokka'
        self.phispy = 'PhiSpy.py'
        self.phanotate = 'phanotate.py'
        self.cdHit = 'cd-hit'
        self.rundbcan = 'run_dbcan.py'
        self.hmmsearch = 'hmmsearch'
        self.rpsblast = 'rpsblast'
        self.deeptmhmm = 'biolib run DTU/DeepTMHMM'
        self.signal = 'signalp6'
        self.DBSCAN_SWA = 'python'


    def run(self, cmd, wkdir=None):
        sys.stderr.write("Running %s ...\n" % cmd)
        p = Popen(cmd, shell=True, cwd=wkdir)
        p.wait()
        return p.returncode

    def run_prokka(self, fastain, fastaout, prefix, type_annotation):
        cmd = '%s %s -o %s --prefix %s --kingdom %s -force' % (self.prokka, fastain, fastaout,prefix,type_annotation)
        return cmd

    def run_phispy(self, gbk_input, out, profix, phage_genes):
        cmd = '%s %s -o %s -p %s --threads 8 --phage_genes %s' % (self.phispy, gbk_input, out, profix, phage_genes)
        return cmd

    def run_phanotate(self, inputfile, out):
        cmd = '%s %s -o %s' % (self.phanotate, inputfile, out)
        return cmd

    def run_cdhit(self,inputfile, out, cutoff):
        cmd = '%s -i %s -o %s -c %s -M 0' % (self.cdHit, inputfile, out, cutoff)
        return cmd

    def scan_dbscan(self,inputfile, out, db):
        cmd = '%s %s protein -t hmmer --out_dir %s --db_dir %s' % (self.rundbcan, inputfile, out, db)
        return cmd

    def run_hmmsearch(self,tblout, e_val, hmm, inputfile):
        cmd = '%s --tblout %s -E %s --cpu 2 %s %s' % (self.hmmsearch, tblout, e_val, hmm, inputfile)
        return cmd
        
    def run_hmmsearch_2(self,out, e_val, hmm, inputfile):
        cmd = '%s --domtblout %s -E %s --cpu 2 %s %s' % (self.hmmsearch, out, e_val, hmm, inputfile)
        return cmd
        
    def run_deeptmhmm(self,fa):
        cmd = '%s --fasta %s' % (self.deeptmhmm,fa)
        return cmd
        
    def run_rpsblast(self,query,evalue,out,db):
        cmd = '%s -query %s -outfmt 6 -evalue %s -out %s -db %s' % (self.rpsblast, query,evalue,out,db)
        return cmd
        
    def run_DBSCAN_SWA(self,path_arg,inp,out,prefix):
        cmd = '%s %s --input %s --output %s --prefix %s' % (self.DBSCAN_SWA,path_arg,inp,out,prefix)
        return cmd
    
    def run_signal(self,fa,out):
        cmd = '%s --fastafile %s --output_dir %s' % (self.signal,fa,out)
        return cmd


def fasta2dict_2(fasta_name):
    with open(fasta_name) as fa:
        fa_dict = {}
        for line in fa:
            line = line.replace('\n', '')
            if line.startswith('>'):
              seq_name = line[1::].strip()
              fa_dict[seq_name] = ''
            else:
              fa_dict[seq_name] += line.replace('\n', '')
    return fa_dict



def main(path, ref):
    tl = tools()
    dic_fa = {}
    with open(path) as f:
        lines = f.readlines()
        first_line = lines[0]
        if first_line.startswith('>'):
            cmd_9 = tl.run_signal(path,'./signaltmp')
            tl.run(cmd_9)
            
            dic_fa = fasta2dict_2(path)
        else:
            state = 'N'
    f.close()
    
    with open('./putative_lysins.fa','w') as w:
      for key in dic_fa:
         line = '>' + key + '\n' + dic_fa[key] + '\n'
         w.write(line)
    w.close()
    
    f1 = open('./molecular_weight.txt')
    f2 = open('./putative_lysin_domain_info.csv')
    
    f1 = open('./molecular_weight.txt')
    f2 = open('./putative_lysin_domain_info.csv')
    
    
    with open('./MW_Length.txt', 'w') as w1:
      for i in f1:
        name = i.strip().split('\t')[0]
        mw = i.strip().split('\t')[1]
        if name in dic_fa.keys():
          line = name + '\t' + mw + '\t' + str(len(dic_fa[name])) + '\n'
          w1.write(line)
    w1.close()
    
    # os.system("sed -i '$d' %s" % ('/home/runzeli/rzli/zy/result/MW_Length.txt'))   
    
    Domain_Info_dict = {}
    with open('./Domain_Info.txt', 'w') as w2:
      for line in f2:
          orf_id = line.strip().split(',')[0]
          info_lis = []
          info = '&'.join(line.strip().split(',')[1:])
          info_lis.append(info)
          if orf_id in Domain_Info_dict:
            Domain_Info_dict[orf_id].append(info)
          else:
            info_lis.append(info)
            Domain_Info_dict[orf_id] = info_lis
      for key in Domain_Info_dict:
          list_tmp = list(set(Domain_Info_dict[key]))
          line = key + '\t' + ';'.join(list_tmp) + '\n'
          w2.write(line)
    w2.close()
              
    # os.system("sed -i '$d' %s" % ('/home/runzeli/rzli/zy/result/Domain_Info.txt'))
    
    
    f1 = open('./MW_Length.txt')
    f2 = open('./Domain_Info.txt')
    f3 = open('./signaltmp/output.gff3')
    
    
    dic_info = {}
    for lines in f1:
      line = lines.strip().split('\t')
      id_1 = line[0]
      mw = line[1]
      length = line[2]
      mw_length = []
      mw_length.append(mw)
      mw_length.append(length)
      dic_info[id_1] = mw_length
      
    
    for lines in f2:
      line = lines.strip().split('\t')
      id_2 = line[0]
      pf = line[1]
      if id_2 in dic_info.keys():
        dic_info[id_2].append(pf)
    
    a = []
    b = []
    for lines in f3:
      if lines[0] != "#":
        line = lines.strip().split('\t')
        id_3 = line[0]
        if float(line[5]) > 0.5:
          li = line[0] + ':' + line[3] + '-' + line[4]
          print(li)
          if id_3 in dic_info.keys():
            dic_info[id_3].append(li)
            a.append(id_3)
    
    for key in dic_info:
      b.append(key)    
    c = list(set(b).difference(set(a)))
    
    for i in c:
      dic_info[i].append('NULL')
            
    print(dic_info)
    
    if ref != '':
      first_dict = SeqIO.to_dict(SeqIO.parse(open('./putative_lysins.fa'),'fasta'))
      os.chdir(curr_dir)
      ref_lysins = os.path.abspath(str(ref))
      second_dict = SeqIO.to_dict(SeqIO.parse(open(ref_lysins),'fasta'))
      os.chdir(Args.workdir)
      
      dic_ref = {}
      # 两个fasta文件中的序列两两比较：
      for t1 in first_dict:
        t_len = len(first_dict[t1].seq)
        for t2 in second_dict:
          global_align = pw2.align.globalxx(first_dict[t1].seq, second_dict[t2].seq)
          matched = global_align[0][2]
          percent_match = (matched / t_len) * 100
          
          if t1 not in dic_ref.keys():
            score = []
            score.append(t2 + ':' + str(round(percent_match,2)))
            dic_ref[t1] = score
          elif t1 in dic_ref.keys():
            dic_ref[t1].append(t2 + ':' + str(round(percent_match,2)))
  
  
      with open('./putative_lysins_info.txt','w') as w:
        line = 'ID' + '\t' + 'MW' + '\t' + 'Length' + '\t' + 'Domains' + '\t' + 'Signalp' + '\t' + 'Reference similarity' + '\n'
        w.write(line)
        for key in dic_info:
          line = key + '\t' + '\t'.join(dic_info[key][0:2]) + '\t' + dic_info[key][2] + '\t' + dic_info[key][-1] + '\t' + '\t'.join(dic_ref[key]) + '\n'
          w.write(line)
      w.close()
      
              
    elif ref == '': 
      with open('./putative_lysins_info.txt','w') as w:
        line = 'ID' + '\t' + 'MW' + '\t' + 'Length' + '\t' + 'Domains' + '\t' + 'Signalp' + '\n'
        w.write(line)
        for key in dic_info:
          line = key + '\t' + '\t'.join(dic_info[key][0:2]) + '\t' + dic_info[key][2] + '\t' + dic_info[key][-1] + '\n'
          w.write(line)
      w.close()
  
  
if __name__ == "__main__":
    main('./rpsblast_cdhit.fasta', ref = '')
