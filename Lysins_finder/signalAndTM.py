#!/usr/bin/env python

import argparse, os
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
from Bio import SeqIO
from Bio import AlignIO
from Bio import pairwise2 as pw2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils.ProtParam import ProteinAnalysis
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


def Split_fa_rps(fasta_name,tot, num_1, num_2):    
    dict = fasta2dict(fasta_name)
    for i in range(1,num_1 + 1):
      dic = dict_slice(dict, int(str(i) + '00') - 100, int(str(i) + '00'))
      with open('./rpsblast_cdhit-' + str(i) + '00.fasta','w') as w:
        for key in dic:
          line = key + '\n' + dic[key] + '\n'
          w.write(line)
      w.close()
    
    if num_2 != 0:
      with open('./rpsblast_cdhit-' + str(int(str(num_1 + 1) + '00')) + '.fasta','w') as w:
        dic = dict_slice(dict, int(str(num_1) + '00'), int(str(num_1) + '00') + int(num_2))
        for key in dic:
          line = key + '\n' + dic[key] + '\n'
          w.write(line)
      w.close()


def Domain_filter_hmmer_withoutRef(Domain_location_dict, ident, coverage, over_lap):
    Domain_location_filter_dict = {}
    Domain_list_get = []
    for item in Domain_location_dict.items():
        key_data = item[0]
        Domain_list = item[1]
        Domain_list.sort(key=operator.itemgetter(2))
        start_initial = 0
        ii_keep = 0
        Domain_filter_list = []
        for ii in range(len(Domain_list)):
            if int(Domain_list[ii][2]) >= start_initial:
                Domain_filter_list.append((Domain_list[ii][0], Domain_list[ii][1], Domain_list[ii][2], 
                                           Domain_list[ii][3], Domain_list[ii][4], Domain_list[ii][5], 
                                           Domain_list[ii][6], Domain_list[ii][7], Domain_list[ii][8], 
                                           Domain_list[ii][9], Domain_list[ii][10], Domain_list[ii][11]))
                start_initial = Domain_list[ii][3]
                ii_keep = ii
            elif int(Domain_list[ii][2]) < start_initial and int(Domain_list[ii][3]) > start_initial and float(
                int(Domain_list[ii][3]) - start_initial) / float(
                int(Domain_list[ii][3]) - Domain_list[ii][2]) > over_lap / 100 and float(
                start_initial - int(Domain_list[ii][2])) / float(int(Domain_list[ii][9])) < (100 - over_lap) / 100:
                Domain_filter_list.append((Domain_list[ii][0], Domain_list[ii][1], Domain_list[ii][2], 
                                           Domain_list[ii][3], Domain_list[ii][4], Domain_list[ii][5], 
                                           Domain_list[ii][6], Domain_list[ii][7], Domain_list[ii][8], 
                                           Domain_list[ii][9], Domain_list[ii][10], Domain_list[ii][11]))
                start_initial = Domain_list[ii][3]
                ii_keep = ii
            else:
                if float(Domain_list[ii][5]) > float(Domain_list[ii_keep][5]):
                    sss = (Domain_list[ii][0], Domain_list[ii][1], Domain_list[ii][2], 
                           Domain_list[ii][3], Domain_list[ii][4], Domain_list[ii][5], 
                           Domain_list[ii][6], Domain_list[ii][7], Domain_list[ii][8], 
                           Domain_list[ii][9], Domain_list[ii][10], Domain_list[ii][11])
                    if sss in Domain_filter_list:
                        Domain_filter_list.remove(sss)
                    Domain_filter_list.append(((Domain_list[ii][0], Domain_list[ii][1], Domain_list[ii][2], 
                                                Domain_list[ii][3], Domain_list[ii][4], Domain_list[ii][5], 
                                                Domain_list[ii][6], Domain_list[ii][7], Domain_list[ii][8], 
                                                Domain_list[ii][9], Domain_list[ii][10], Domain_list[ii][11])))
                    start_initial = Domain_list[ii][3]
                    ii_keep = ii
                else:
                    continue
        Domain_filter_list_use = list(set(Domain_filter_list))
        for j in Domain_filter_list_use:
            if j not in Domain_list_get:
                Domain_list_get.append(j[0])
                Domain_list_get = list(set(Domain_list_get))
        Domain_location_filter_dict[key_data] = Domain_filter_list_use
    Domain_location_use_dict = {}
    for kk in Domain_location_filter_dict.items():
        ID_filter = kk[0]
        Domain_list = kk[1]
        for i in Domain_list:
            if float(i[5]) >= float(ident) / 100 and float(i[6]) >= float(coverage) / 100:  
                Domain_location_use_dict.setdefault(ID_filter, []).append(i)

    return Domain_location_use_dict


def Domain_filter_hmmer_withRef(Domain_location_dict, ident, coverage, over_lap):
    Domain_location_filter_dict = {}
    Domain_list_get = []
    for item in Domain_location_dict.items():
        key_data = item[0]
        Domain_list = item[1]
        Domain_list.sort(key=operator.itemgetter(2))
        start_initial = 0
        ii_keep = 0
        Domain_filter_list = []
        for ii in range(len(Domain_list)):
            if int(Domain_list[ii][2]) >= start_initial:
                Domain_filter_list.append((Domain_list[ii][0], Domain_list[ii][1], Domain_list[ii][2], 
                                           Domain_list[ii][3], Domain_list[ii][4], Domain_list[ii][5], 
                                           Domain_list[ii][6], Domain_list[ii][7], Domain_list[ii][8], 
                                           Domain_list[ii][9], Domain_list[ii][10], Domain_list[ii][11],
                                           Domain_list[ii][12]))
                start_initial = Domain_list[ii][3]
                ii_keep = ii
            elif int(Domain_list[ii][2]) < start_initial and int(Domain_list[ii][3]) > start_initial and float(
                int(Domain_list[ii][3]) - start_initial) / float(
                int(Domain_list[ii][3]) - Domain_list[ii][2]) > over_lap / 100 and float(
                start_initial - int(Domain_list[ii][2])) / float(int(Domain_list[ii][9])) < (100 - over_lap) / 100:
                Domain_filter_list.append((Domain_list[ii][0], Domain_list[ii][1], Domain_list[ii][2], 
                                           Domain_list[ii][3], Domain_list[ii][4], Domain_list[ii][5], 
                                           Domain_list[ii][6], Domain_list[ii][7], Domain_list[ii][8], 
                                           Domain_list[ii][9], Domain_list[ii][10], Domain_list[ii][11],
                                           Domain_list[ii][12]))
                start_initial = Domain_list[ii][3]
                ii_keep = ii
            else:
                if float(Domain_list[ii][5]) > float(Domain_list[ii_keep][5]):
                    sss = (Domain_list[ii][0], Domain_list[ii][1], Domain_list[ii][2], 
                           Domain_list[ii][3], Domain_list[ii][4], Domain_list[ii][5], 
                           Domain_list[ii][6], Domain_list[ii][7], Domain_list[ii][8], 
                           Domain_list[ii][9], Domain_list[ii][10], Domain_list[ii][11],
                           Domain_list[ii][12])
                    if sss in Domain_filter_list:
                        Domain_filter_list.remove(sss)
                    Domain_filter_list.append(((Domain_list[ii][0], Domain_list[ii][1], Domain_list[ii][2], 
                                                Domain_list[ii][3], Domain_list[ii][4], Domain_list[ii][5], 
                                                Domain_list[ii][6], Domain_list[ii][7], Domain_list[ii][8], 
                                                Domain_list[ii][9], Domain_list[ii][10], Domain_list[ii][11],
                                                Domain_list[ii][12])))
                    start_initial = Domain_list[ii][3]
                    ii_keep = ii
                else:
                    continue
        Domain_filter_list_use = list(set(Domain_filter_list))
        for j in Domain_filter_list_use:
            if j not in Domain_list_get:
                Domain_list_get.append(j[0])
                Domain_list_get = list(set(Domain_list_get))
        Domain_location_filter_dict[key_data] = Domain_filter_list_use
    Domain_location_use_dict = {}
    for kk in Domain_location_filter_dict.items():
        ID_filter = kk[0]
        Domain_list = kk[1]
        for i in Domain_list:
            if float(i[5]) >= float(ident) / 100 and float(i[6]) >= float(coverage) / 100:  
                Domain_location_use_dict.setdefault(ID_filter, []).append(i)

    return Domain_location_use_dict


def main_hmmer(file, method, hmmer_coverage, hmmer_over_lap, hmmer_accuracy, ref, cbd, ead, wkdir):
    tl = tools()
    filename = os.path.basename(file)
    lis = filename.split('.')[:-1]
    prefix_file = '.'.join(lis)
    
    tot = sub.getoutput("grep '>' %s | wc -l" % ('./' + filename))
    
    if os.path.isdir(wkdir) == True:
        pass
    else:
        os.mkdir(wkdir)
        
    os.chdir(wkdir)
    if int(tot) > 100:
        num_1 = int(tot)//100
        num_2 = int(tot)%100
        Split_fa('./' + filename, tot, num_1, num_2)
      
        for i in range(1, int(num_1) + 1):
           time_sleep = random.uniform(60, 180)
           time.sleep(time_sleep)
           cmd_8 = tl.run_deeptmhmm('./' + prefix_file + '-' + str(i) + '00.fasta')
           tl.run(cmd_8)
           
        os.system('cat ./biolib_results/predicted_topologies.3line* > ./biolib_results/predicted_topologies.line')
        remove_TMhelix('./biolib_results/predicted_topologies.line','./pfam_EAD_cdhit.fasta','./putative_lysins.fa')
      
    else:
        cmd_8 = tl.run_deeptmhmm('./' + filename)
        tl.run(cmd_8)
        remove_TMhelix('./biolib_results/predicted_topologies.3line','./pfam_EAD_cdhit.fasta','./putative_lysins.fa')


    dic_fa = {}
    with open('./putative_lysins.fa') as f:
      lines = f.readlines()
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
        
      domain_type = {}
      f_CBD = open(cbd)
      for i in f_CBD:
        item = i.strip()
        domain_type[item] = 'CBD'
      
      f_EAD = open(ead)
      for i in f_EAD:
        item = i.strip()
        domain_type[item] = 'EAD'
      
      for lines in f2:
        line = lines.strip().split('\t')
        id_2 = line[0]
        pf = line[1] + '&' + line[2] + '&' + domain_type[line[2].split('(Length')[0]] + '&' + line[3] + '&' + line[4]
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
        ref_lysins = os.path.abspath(str(ref))
        second_dict = SeqIO.to_dict(SeqIO.parse(open(ref_lysins),'fasta'))
        
        dic_ref = {}
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

          
        with open('./putative_lysins_info_tmp.txt','w') as w:
          line = 'ID' + '\t' + 'MW' + '\t' + 'Length' + '\t' + 'Domains' + '\t' + 'Signalp' + '\t' + 'Reference similarity' + '\n'
          w.write(line)
          for key in dic_info:
            line = key + '\t' + '\t'.join(dic_info[key][0:2]) + '\t' + ';'.join(dic_info[key][2:len(dic_info[key])-1]) + '\t' + dic_info[key][-1] + '\t' + '&'.join(dic_ref[key]) + '\n'
            w.write(line)
        w.close()
        
        f = open('./putative_lysins_info_tmp.txt')
        next(f)
        putative_lysin_dict = {}
        for i in f:
          id = i.strip().split('\t')[0]
          info = i.strip().split('\t')[3].split(';')
          print(info)
          for j in info:
            j_name = j.split('&')[0]
            j_id = j.split('&')[1]
            j_type = j.split('&')[2]
            j_ident = j.split('&')[3]
            j_start = int(j.split('&')[4].split('-')[0])
            j_end = int(j.split('&')[4].split('-')[1])
            j_mw = i.strip().split('\t')[1]
            j_length = i.strip().split('\t')[2]
            Domain_len = int(j.split('&')[0].split('(Length:')[1].replace(')',''))
            align_percent = '%.2f' % (float(j_end - j_start + 1) / float(Domain_len) * 100)
            signalp = i.strip().split('\t')[4]
            ref = i.strip().split('\t')[5]
            protein_length = i.strip().split('\t')[2]
          
            putative_lysin_dict.setdefault(id, []).append(
                              (j_id, j_name, j_start,
                               j_end, j_type,
                               j_ident, align_percent,
                               id, j_mw, Domain_len,signalp, protein_length, ref))
          
        coverage = hmmer_coverage
        over_lap = hmmer_over_lap
        ident = hmmer_accuracy
        Domain_location_use_dict = Domain_filter_hmmer_withRef(putative_lysin_dict, ident, coverage, over_lap)
        
        fa_id = []
        with open('./putative_lysins_info.txt','w') as w:
          line = 'ID' + '\t' + 'MW' + '\t' + 'Length' + '\t' + 'Domains' + '\t' + 'Signalp' + '\t' + 'Reference similarity' + '\n'
          w.write(line)
          for key in Domain_location_use_dict:
            lis = []
            lis_2 = []
            if len(Domain_location_use_dict[key]) == 1:
              inv = str(Domain_location_use_dict[key][0][2]) + '-' + str(Domain_location_use_dict[key][0][3])
              lis = [Domain_location_use_dict[key][0][0], Domain_location_use_dict[key][0][1], Domain_location_use_dict[key][0][4], Domain_location_use_dict[key][0][5], inv]
              if 'EAD' in line:
                line = key + '\t' + Domain_location_use_dict[key][0][8] + '\t' + Domain_location_use_dict[key][0][11] + '\t' + '&'.join(lis) + '\t' + Domain_location_use_dict[key][0][10] + '\t' + Domain_location_use_dict[key][0][12] + '\n'
                w.write(line)
                fa_id.append(key)
            elif len(Domain_location_use_dict[key]) > 1:
              for j in Domain_location_use_dict[key]:
                inv = str(j[2]) + '-' + str(j[3])
                lis = [j[0], j[1], j[4], j[5], inv]
                lis_2.append('&'.join(lis))
              
              line = key + '\t' + Domain_location_use_dict[key][0][8] + '\t' + Domain_location_use_dict[key][0][11] + '\t' + ';'.join(lis_2) + '\t' + Domain_location_use_dict[key][0][10] + '\t' + Domain_location_use_dict[key][0][12] + '\n'
              if 'EAD' in line:
                w.write(line)
                fa_id.append(key)
        w.close()
        
        os.system('mv ./putative_lysins.fa ./putative_lysins_tmp.fa')
        
        putative_fa = fasta2dict_2('./putative_lysins_tmp.fa')
        
        with open('./putative_lysins.fa', 'w') as w:
          for key in putative_fa:
            if key in fa_id:
              line = '> ' + key + '\n' + putative_fa[key] + '\n'
              w.write(line)
        w.close()
        
                
      elif ref == '':
        print('aaaaa')
        
        with open('./putative_lysins_info_tmp.txt','w') as w:
          line = 'ID' + '\t' + 'MW' + '\t' + 'Length' + '\t' + 'Domains' + '\t' + 'Signalp' + '\n'
          w.write(line)
          for key in dic_info:
            line = key + '\t' + '\t'.join(dic_info[key][0:2]) + '\t' + ';'.join(dic_info[key][2:len(dic_info[key])-1]) + '\t' + dic_info[key][-1] + '\n'
            w.write(line)
        w.close()
        
        f = open('./putative_lysins_info_tmp.txt')
        next(f)
        putative_lysin_dict = {}
        for i in f:
          id = i.strip().split('\t')[0]
          info = i.strip().split('\t')[3].split(';')
          print(info)
          for j in info:
            j_name = j.split('&')[0]
            j_id = j.split('&')[1]
            j_type = j.split('&')[2]
            j_ident = j.split('&')[3]
            j_start = int(j.split('&')[4].split('-')[0])
            j_end = int(j.split('&')[4].split('-')[1])
            j_mw = i.strip().split('\t')[1]
            j_length = i.strip().split('\t')[2]
            Domain_len = int(j.split('&')[0].split('(Length:')[1].replace(')',''))
            align_percent = '%.2f' % (float(j_end - j_start + 1) / float(Domain_len) * 100)
            signalp = i.strip().split('\t')[4]
            protein_length = i.strip().split('\t')[2]
          
            putative_lysin_dict.setdefault(id, []).append(
                              (j_id, j_name, j_start,
                               j_end, j_type,
                               j_ident, align_percent,
                               id, j_mw, Domain_len,signalp, protein_length))
          
        coverage = hmmer_coverage
        over_lap = hmmer_over_lap
        ident = hmmer_accuracy
        Domain_location_use_dict = Domain_filter_hmmer_withoutRef(putative_lysin_dict, ident, coverage, over_lap)
        
        fa_id = []
        with open('./putative_lysins_info.txt','w') as w:
          line = 'ID' + '\t' + 'MW' + '\t' + 'Length' + '\t' + 'Domains' + '\t' + 'Signalp' + '\n'
          w.write(line)
          for key in Domain_location_use_dict:
            lis = []
            lis_2 = []
            if len(Domain_location_use_dict[key]) == 1:
              inv = str(Domain_location_use_dict[key][0][2]) + '-' + str(Domain_location_use_dict[key][0][3])
              lis = [Domain_location_use_dict[key][0][0], Domain_location_use_dict[key][0][1], Domain_location_use_dict[key][0][4], Domain_location_use_dict[key][0][5], inv]
              line = key + '\t' + Domain_location_use_dict[key][0][8] + '\t' + Domain_location_use_dict[key][0][11] + '\t' + '&'.join(lis) + '\t' + Domain_location_use_dict[key][0][10] + '\n'
              if 'EAD' in line:
                w.write(line)
                fa_id.append(key)
            elif len(Domain_location_use_dict[key]) > 1:
              for j in Domain_location_use_dict[key]:
                inv = str(j[2]) + '-' + str(j[3])
                lis = [j[0], j[1], j[4], j[5], inv]
                lis_2.append('&'.join(lis))
              
              line = key + '\t' + Domain_location_use_dict[key][0][8] + '\t' + Domain_location_use_dict[key][0][11] + '\t' + ';'.join(lis_2) + '\t' + Domain_location_use_dict[key][0][10] + '\n'
              if 'EAD' in line:
                w.write(line)
                fa_id.append(key)
        w.close()
        
        os.system('mv ./putative_lysins.fa ./putative_lysins_tmp.fa')
        
        putative_fa = fasta2dict_2('./putative_lysins_tmp.fa')
        
        with open('./putative_lysins.fa', 'w') as w:
          for key in putative_fa:
            if key in fa_id:
              line = '> ' + key + '\n' + putative_fa[key] + '\n'
              w.write(line)
        w.close()
        

def main_rpsblast(file, method, wkdir, ref):
    tl = tools()
    filename = os.path.basename(file)
    lis = filename.split('.')[:-1]
    prefix_file = '.'.join(lis)
    
    tot = sub.getoutput("grep '>' %s | wc -l" % ('./' + filename))

    if int(tot) > 100:
        num_1 = int(tot)//100
        num_2 = int(tot)%100
        Split_fa_rps('./' + filename, tot, num_1, num_2)
      
        for i in range(1, int(num_1) + 1):
           time_sleep = random.uniform(60, 180)
           time.sleep(time_sleep)
           cmd_8 = tl.run_deeptmhmm('./' + prefix_file + '-' + str(i) + '00.fasta')
           tl.run(cmd_8)
           
        os.system('cat ./biolib_results/predicted_topologies.3line* > ./biolib_results/predicted_topologies.line')
        remove_TMhelix('./biolib_results/predicted_topologies.line','./rpsblast_cdhit.fasta','./putative_lysins.fa')
      
    else:
        cmd_8 = tl.run_deeptmhmm('./' + filename)
        tl.run(cmd_8)
        remove_TMhelix('./biolib_results/predicted_topologies.3line','./rpsblast_cdhit.fasta','./putative_lysins.fa')
      
      
    dic_fa = {}
    with open('./putative_lysins.fa') as f:
        lines = f.readlines()
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
    f2 = open('./putative_lysin_domain_info.csv')
    
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
        ref_lysins = os.path.abspath(str(ref))
        second_dict = SeqIO.to_dict(SeqIO.parse(open(ref_lysins),'fasta'))
        
        dic_ref = {}
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
    parser = argparse.ArgumentParser(description="Lysin annotation")
    parser.add_argument("-f", "--file", required=True, type=str, help="input file")
    parser.add_argument("-wkdir", "--workdir", required=True, type=str, help="work directory")
    parser.add_argument("-rlc", "--reported_lysin_CBD", required=False, type=str, help="reported lysin CBD structures(hmm files,hmmer)")
    parser.add_argument("-rle", "--reported_lysin_EAD", required=False, type=str, help="reported lysin EAD structures(hmm files,hmmer)")
    parser.add_argument("-hcov", "--hmmer_coverage", required=False, default=80, type=float, help="hmmer region coverage(hmmer)")
    parser.add_argument("-hacc", "--hmmer_accuracy", required=False, default=50, type=float, help="hmmer accuracy(hmmer)")
    parser.add_argument("-hol", "--hmmer_over_lap", required=False, default=80, type=float, help="hmmer cutoff of overlap in the same region(hmmer)")
    parser.add_argument("-m", "--method", default='hmmer', required=True, type=str, help="searching method 'hmmer' or 'rpsblast'")
    parser.add_argument("-r", "--ref", default='', required=False, type=str, help="reference lysins sequences")
    Args = parser.parse_args()
    
    if Args.method == 'hmmer':
      file = Args.file
      method = Args.method
      reported_lysin_CBD_suffix = Args.reported_lysin_CBD
      #'/home/user/deeplysin/database/hmm/lysin_reported_CBD.txt'
      reported_lysin_EAD_suffix = Args.reported_lysin_EAD
      #'/home/user/deeplysin/database/hmm/lysin_reported_EAD.txt'
      hmmer_coverage = Args.hmmer_coverage
      hmmer_over_lap = Args.hmmer_over_lap
      hmmer_accuracy = Args.hmmer_accuracy
      ref_seq = Args.ref
      wkdir = Args.workdir
      if os.path.isdir(wkdir) == True:
        pass
      else:
        os.mkdir(wkdir)
      os.system('cp %s %s' % (file, wkdir))
      os.chdir(wkdir)
      
      if ref_seq != '':
        main_hmmer(file, method = method, ref = ref_seq, cbd = reported_lysin_CBD_suffix, ead = reported_lysin_EAD_suffix, hmmer_coverage = hmmer_coverage, hmmer_over_lap = hmmer_over_lap, hmmer_accuracy = hmmer_accuracy, wkdir = wkdir)
      elif ref_seq == '':
        main_hmmer(file, method = method, ref = '', cbd = reported_lysin_CBD_suffix, ead = reported_lysin_EAD_suffix, hmmer_coverage = hmmer_coverage, hmmer_over_lap = hmmer_over_lap, hmmer_accuracy = hmmer_accuracy, wkdir = wkdir)
    
    
    elif Args.method == 'rpsblast':
      file = Args.file
      method = Args.method
      ref_seq = Args.ref
      wkdir = Args.workdir
      if os.path.isdir(wkdir) == True:
        pass
      else:
        os.mkdir(wkdir)
      os.system('cp %s %s' % (file, wkdir))
      os.chdir(wkdir)
      
      if ref_seq != '':
        main_rpsblast(file, method = method, ref = ref_seq, wkdir = wkdir)
      elif ref_seq == '':
        main_rpsblast(file, method = method, ref = '', wkdir = wkdir)