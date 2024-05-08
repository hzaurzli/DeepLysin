#!/usr/bin/env python

import argparse
import os,sys,re,time
import random
import subprocess as sub
from subprocess import *
import subprocess as sub
import glob
import shutil
import biolib
from Bio import SeqIO
from Bio import AlignIO
from Bio import pairwise2 as pw2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils.ProtParam import ProteinAnalysis


class tools:
    def __init__(self):
        self.prokka = 'prokka'
        self.phispy = 'PhiSpy.py'
        self.phanotate = 'phanotate.py'
        self.cdHit = 'cd-hit'
        self.rundbcan = 'run_dbcan.py'
        self.hmmsearch = 'hmmsearch'
        self.deeptmhmm = 'biolib run DTU/DeepTMHMM'
        self.signal = 'signalp6'


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
        
    def run_signal(self,fa,out):
        cmd = '%s --fastafile %s --output_dir %s' % (self.signal,fa,out)
        return cmd


def fasta2dict(fasta_name):
    with open(fasta_name) as fa:
        fa_dict = {}
        for line in fa:
            line = line.replace('\n', '')
            if line.startswith('>'):
              seq_name = line[0:]
              fa_dict[seq_name] = ''
            else:
              fa_dict[seq_name] += line.replace('\n', '')
    return fa_dict


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



def detete_TMhelix(cdhit_fasta,cazyme_pfam_TMhelix):
    input_file = open(cdhit_fasta, "r")
    info_file = open(cazyme_pfam_TMhelix, "r")

    lines = info_file.readlines()
    info_short_file = open(r"./all_protein_final_tmhmm_shortout.txt", 'w')
    content = "#"
    for line in lines:
        if line.strip()[0] != content:
            info_short_file.write(line)
    info_file.close()
    info_short_file.close()

    info_short_file = open(r"./all_protein_final_tmhmm_shortout.txt", 'r')
    info_list = []
    for i in info_short_file:
        data = i.strip().split('\t')
        gen_id = str(data[0])
        if gen_id not in info_list:
            info_list.append(gen_id)
    info_short_file.close()

    info_short_file = open(r"./all_protein_final_tmhmm_shortout.txt", 'r')
    dele_list = []
    for j in info_short_file:
        data_dele = j.strip().split('\t')
        gen_id_dele = str(data_dele[0])
        protein_type = str(data_dele[2])
        dele_list.append((gen_id_dele, protein_type))
    info_short_file.close()
    for k in dele_list:
        gen_id_dele_last = str(k[0])
        protein_type_dele = str(k[1])
        if (protein_type_dele == "TMhelix") and (gen_id_dele_last in info_list):
            info_list.remove(gen_id_dele_last)

    out_file = open("putative_lysins.fa", "a")
    for record in SeqIO.parse(input_file, 'fasta'):
        Contig_ID = record.id
        Desp = record.description
        for i in info_list:
            gen_id = i
            if gen_id == Contig_ID:
                gene_seq = record.seq
                element_record = SeqRecord(gene_seq, id='', description=Desp)
                SeqIO.write(element_record, out_file, "fasta")
    input_file.close()
    out_file.close()


def dict_slice(adict, start, end):
    keys = adict.keys()
    dict_slice = {}
    for k in list(keys)[start:end]:
      dict_slice[k] = adict[k]
    return dict_slice
  

def Split_fa(fasta_name,tot, num_1, num_2):    
    dict = fasta2dict(fasta_name)
    for i in range(1,num_1 + 1):
      dic = dict_slice(dict, int(str(i) + '00') - 100, int(str(i) + '00'))
      with open('./pfam_EAD_cdhit-' + str(i) + '00.fasta','w') as w:
        for key in dic:
          line = key + '\n' + dic[key] + '\n'
          w.write(line)
      w.close()
    
    if num_2 != 0:
      with open('./pfam_EAD_cdhit-' + str(int(str(num_1 + 1) + '00')) + '.fasta','w') as w:
        dic = dict_slice(dict, int(str(num_1) + '00'), int(str(num_1) + '00') + int(num_2))
        for key in dic:
          line = key + '\n' + dic[key] + '\n'
          w.write(line)
      w.close()
    
    
def remove_TMhelix(TMhelix_path,fa,fa_out):
  f = open(TMhelix_path)
  lis = []
  for i in f:
    if i.startswith('>'):
      type = i.strip().split(' ')[-1]
      if type != 'TM':
        id = i[1::].strip().split(' ')[0]
        lis.append(id)

  fi = fasta2dict(fa)
  if len(lis) == 0:
      with open(fa_out,'w') as w:
          line = 'All lysins have TMhelix'
          w.write(line)
      w.close()
  else:
      with open(fa_out,'w') as w:
        for key in fi:
          for id in lis:
            if id in key:
              line = key + '\n' + fi[key] + '\n'
              w.write(line)
      w.close()


if __name__ == "__main__":

  tl = tools()
  # step 12 remove TMhelix
  tot = sub.getoutput("grep '>' %s | wc -l" % ('./pfam_EAD_cdhit.fasta'))
  
  if int(tot) > 100:
      num_1 = int(tot)//100
      num_2 = int(tot)%100
      Split_fa('./pfam_EAD_cdhit.fasta', tot, num_1, num_2)
    
      for i in range(1, int(num_1) + 2):
         time_sleep = random.uniform(60, 180)
         time.sleep(time_sleep)
         cmd_8 = tl.run_deeptmhmm('./pfam_EAD_cdhit-' + str(i) + '00.fasta')
         print('pfam_EAD_cdhit-' + str(i) + '00.fasta')
         tl.run(cmd_8)
         
      os.system('cat ./biolib_results/predicted_topologies.3line* > ./biolib_results/predicted_topologies.line')
      remove_TMhelix('./biolib_results/predicted_topologies.line','./pfam_EAD_cdhit.fasta','./putative_lysins.fa')
    
  else:
      cmd_8 = tl.run_deeptmhmm('./pfam_EAD_cdhit.fasta')
      tl.run(cmd_8)
      remove_TMhelix('./biolib_results/predicted_topologies.3line','./pfam_EAD_cdhit.fasta','./putative_lysins.fa')
  
  
  dic_fa = {}
  with open('./putative_lysins.fa') as f:
    lines = f.readlines()  # 读取所有行
    first_line = lines[0]
    if first_line.startswith('>'):
        state = 'Y'
        cmd_9 = tl.run_signal('./putative_lysins.fa','./signaltmp')
        tl.run(cmd_9)
        
        dic_fa = fasta2dict_2('./putative_lysins.fa')
    else:
        state = 'N'
  f.close()
  
  f1 = open('./molecular_weight.txt')
  f2 = open('./hmmer_out/all_protein.txt')
  
  
  if state == 'Y':
    with open('./MW_Length.txt', 'w') as w1:
      for i in f1:
        name = i.strip().split('\t')[0]
        mw = i.strip().split('\t')[1]
        if name in dic_fa.keys():
          line = name + '\t' + mw + '\t' + str(len(dic_fa[name])) + '\n'
          w1.write(line)
    w1.close()
    
    # os.system("sed -i '$d' %s" % ('/home/runzeli/rzli/zy/result/MW_Length.txt'))
    
    Domain_Info_lis = []
    with open('./Domain_Info.txt', 'w') as w2:
      for line in f2:
        if line[0] != "#" and len(line.split())!=0:
          arr = line.strip().split(" ")
          arr = list(filter(None, arr))
          name = arr[0]
          if name in dic_fa.keys():
            li = arr[0] + '\t' + arr[3] + '(Length:' + arr[5] + ')' + '\t' + arr[4].split('.')[0] + '(Length:' + arr[5] + ')' + '\t' + arr[21] + '\t' + arr[19] + '-' + arr[20] + '\n'
            print(li)
            Domain_Info_lis.append(li)
            
      Domain_Info_lis_new = list(set(Domain_Info_lis))
      for line in Domain_Info_lis_new:
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
      pf = line[1] + '&' + line[2] + '&' + line[3] + '&'  + line[4]
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
    
    
    with open('./putative_lysins_info.txt','w') as w:
      line = 'ID' + '\t' + 'MW' + '\t' + 'Length' + '\t' + 'Domains' + '\t' + 'Signalp' + '\n'
      w.write(line)
      for key in dic_info:
        line = key + '\t' + '\t'.join(dic_info[key][0:2]) + '\t' + ';'.join(dic_info[key][2:len(dic_info[key])-1]) + '\t' + dic_info[key][-1] + '\n'
        w.write(line)
    w.close()
      
          
  else:
    print(state)
