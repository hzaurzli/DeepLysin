#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Lysins_finder.py 
#
#  Copyright 2022 Small runze
#  <small.runze@gmail.com> Small runze
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., HZAU.


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


def prophage_select(prophage_input,fna_input,ppn_out):
    if os.path.exists(prophage_input):
        info_file = open(prophage_input, "r")
        info_list = []
        for i in info_file:
            data = i.strip().split('\t')
            pp_num = data[0]
            contig = data[1]
            info_s = data[2]
            info_e = data[3]
            info_list.append((pp_num, contig, info_s, info_e))

        for record in SeqIO.parse(fna_input, 'fasta'):
            Contig_ID = record.id
            Desp = record.description.split(",")[0]
            for i in info_list:
                pp_num = i[0]
                contig = i[1]
                info_s = i[2]
                info_e = i[3]
                if contig == Contig_ID:
                    file_name_use = ppn_out
                    out_file = open(file_name_use + "_" + str(pp_num) + ".fasta", "a")
                    gene_seq = record.seq[int(info_s) - 1: int(info_e)]
                    element_record = SeqRecord(gene_seq, id=pp_num, description=Desp)
                    SeqIO.write(element_record, out_file, "fasta")
                    out_file.close()
        info_file.close()


def Gene_element_abstract(ppn_out,ppn_fa,ppn_ffn):
    n = 0
    gene_list = []

    list_path = str(ppn_fa).split('/')
    list_path = [i for i in list_path if i != '']
    if len(list_path) == 1:
        file_name = ppn_fa
    else:
        file_name = os.path.basename(ppn_fa)

    ppn_out = open(ppn_out,'r')
    for line in ppn_out:
        if line.startswith('#'):
            pass
        else:
            line_info = line.strip().split("\t")
            Contig_ID = file_name.strip().split(".")[0]
            location_S = int(line_info[0])  # start
            location_E = int(line_info[1])  # end
            if line_info[2] == "-":
                F_R = "R"  # reverse
                location_S = int(line_info[1])
                location_E = int(line_info[0])
            if line_info[2] == "+":
                F_R = "F"
                location_S = int(line_info[0])
                location_E = int(line_info[1])
            gene_list.append((location_S, location_E, F_R))
    ppn_out.close()

    out_file = open(ppn_ffn, "a")
    for record in SeqIO.parse(ppn_fa, "fasta"):
        ID_contig = record.id
        Desp = record.description
        for info in gene_list:
            file_name_1 = file_name
            file_name_2 = file_name_1.split(".")[0]
            if info[2] == "F":
                gene_seq = record.seq[info[0] - 1:info[1]]
                gene_protein = gene_seq.translate()
            elif info[2] == "R":
                gene_seq_ori = record.seq[info[0] - 1:info[1]]
                gene_seq = gene_seq_ori.reverse_complement()
                gene_protein = gene_seq.translate()
            element_ID = file_name_2 + ":" + info[2] + ":" + str(info[0]) + "-" + str(info[1])
            element_record = SeqRecord(gene_protein[0:-1],id=element_ID, description=Desp)
            SeqIO.write(element_record, out_file, "fasta")
    out_file.close()


def molecular_weight(protein_fa,protein_filter_fa,MWU,MWL):
    protein_fa_info = open(protein_fa, "r")
    out_file = open(protein_filter_fa, "a")
    molecular_weight = open("./molecular_weight.txt", "w")
    for record in SeqIO.parse(protein_fa_info, "fasta"):
        ID_contig = record.id
        Seq_use = record.seq
        Desp = record.description
        protein_seq = str(Seq_use)
        if 'X' not in protein_seq and '*' not in protein_seq[:-1]:
            X = ProteinAnalysis(protein_seq)
            MW_cal = "%0.2f" % X.molecular_weight()
            if float(MW_cal) <= float(MWU) and float(MW_cal) >= float(MWL):
                element_record = SeqRecord(Seq_use, id=ID_contig, description=Desp)
                SeqIO.write(element_record, out_file, "fasta")
                molecular_weight.write(ID_contig + "\t" + MW_cal + "\n")

    protein_fa_info.close()
    molecular_weight.close()

def find_cazyme(cdhit_filter,cazy_overview):
    input_file = open(cdhit_filter, "r")
    info_file = open(cazy_overview, "r")
    info_list = []
    for i in info_file:
        data = i.strip().split('\t')
        gen_id = data[0]
        info_list.append(gen_id)

    for record in SeqIO.parse(input_file, 'fasta'):
        Contig_ID = record.id
        Desp = record.description
        for i in info_list:
            gen_id = i
            if gen_id == Contig_ID:
                gene_seq = record.seq
                out_file = open("./all_protein_filter_cazyme.fasta", "a")
                element_record = SeqRecord(gene_seq, id='', description=Desp)
                SeqIO.write(element_record, out_file, "fasta")
                out_file.close()
    input_file.close()
    info_file.close()



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


def find_pfam(cdhit_filter,lyase_list):
    input_file = open(cdhit_filter, "r")
    info_file = open('./hmmer_out/all_protein_filter_hmmer_out.txt', "r")
    info_file_two = open(lyase_list, "r")

    f2 = open(r'./hmmer_out/all_protein_filter_hmmer_out2.txt', 'w')
    for i in info_file:
        line = re.split('\s+', i)
        new_line = ' '.join(line)
        new_line = new_line.strip(' ')
        if new_line[0] != '#':
            f2.write(new_line)
            f2.write("\n")
    info_file.close()
    f2.close()

    info_pfam_list = []
    for i in info_file_two:
        data2 = i.strip()
        pfam_num_reported = str(data2)
        info_pfam_list.append(pfam_num_reported)

    info_file_new = open(r'./hmmer_out/all_protein_filter_hmmer_out2.txt', 'r')
    info_list = []
    for j in info_file_new:
        data1 = j.strip().split(' ')
        gen_id = str(data1[0])
        pfam_num = str(data1[3])
        for k in info_pfam_list:
            if k in pfam_num:
                info_list.append(gen_id)

    out_file = open("./all_protein_pfam_protein.fasta", "a")
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
    info_file_new.close()
    info_file_two.close()
    out_file.close()


def find_pfam_EAD(cdhit_filter,lyase_list):
    input_file = open(cdhit_filter, "r")
    info_file = open('./hmmer_out_EAD/all_protein_filter_hmmer_out_EAD.txt', "r")
    info_file_two = open(lyase_list, "r")

    f2 = open(r'./hmmer_out_EAD/all_protein_filter_hmmer_out2_EAD.txt', 'w')
    for i in info_file:
        line = re.split('\s+', i)
        new_line = ' '.join(line)
        new_line = new_line.strip(' ')
        if new_line[0] != '#':
            f2.write(new_line)
            f2.write("\n")
    info_file.close()
    f2.close()

    info_pfam_list = []
    for i in info_file_two:
        data2 = i.strip()
        pfam_num_reported = str(data2)
        info_pfam_list.append(pfam_num_reported)

    info_file_new = open(r'./hmmer_out_EAD/all_protein_filter_hmmer_out2_EAD.txt', 'r')
    info_list = []
    for j in info_file_new:
        data1 = j.strip().split(' ')
        gen_id = str(data1[0])
        pfam_num = str(data1[3])
        for k in info_pfam_list:
            if k in pfam_num:
                info_list.append(gen_id)

    out_file = open("./all_protein_pfam_protein_EAD.fasta", "a")
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
    info_file_new.close()
    info_file_two.close()
    out_file.close()


def find_pfam_peptidases(cdhit_filter,lyase_list):
    input_file = open(cdhit_filter, "r")
    info_file = open('./hmmer_out_peptidases/all_protein_filter_hmmer_out_peptidases.txt', "r")
    info_file_two = open(lyase_list, "r")

    f2 = open(r'./hmmer_out_peptidases/all_protein_filter_hmmer_out2_peptidases.txt', 'w')
    for i in info_file:
        line = re.split('\s+', i)
        new_line = ' '.join(line)
        new_line = new_line.strip(' ')
        if new_line[0] != '#':
            f2.write(new_line)
            f2.write("\n")
    info_file.close()
    f2.close()

    info_pfam_list = []
    for i in info_file_two:
        data2 = i.strip()
        pfam_num_reported = str(data2)
        info_pfam_list.append(pfam_num_reported)

    info_file_new = open(r'./hmmer_out_peptidases/all_protein_filter_hmmer_out2_peptidases.txt', 'r')
    info_list = []
    for j in info_file_new:
        data1 = j.strip().split(' ')
        gen_id = str(data1[0])
        pfam_num = str(data1[3])
        for k in info_pfam_list:
            if k in pfam_num:
                info_list.append(gen_id)

    out_file = open("./all_protein_pfam_protein_peptidases.fasta", "a")
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
    info_file_new.close()
    info_file_two.close()
    out_file.close()



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
    for i in range(1,num_1):
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
    parser = argparse.ArgumentParser(description="Lysin finder")
    parser.add_argument("-p", "--path", required=True, type=str, help="genome sequnce path")
    parser.add_argument("-t", "--type", required=True, type=str, help="prokka kingdom type")
    parser.add_argument("-c", "--cdhit_cutoff", default=0.95,required=False, type=float, help="cdhit cluster cutoff")
    parser.add_argument("-hc", "--hmmer_cutoff", default=1e-5,required=False, type=float, help="hmmer search cutoff")
    parser.add_argument("-hd", "--hmmer_db", required=True, type=str, help="reported lysin structures hmmer database path")
    parser.add_argument("-rl", "--reported_lysin", required=True, type=str, help="reported lysin structures(hmm files)")
    parser.add_argument("-wkdir", "--workdir", required=True, type=str, help="work directory")
    parser.add_argument("-mu", "--MWU", required=False, default=40000, type=float, help="upper proteins molecular weight")
    parser.add_argument("-ml", "--MWL", required=False, default=10000, type=float, help="lower proteins molecular weight")
    parser.add_argument("-hde", "--hmmer_db_EAD", required=True, type=str, help="EAD hmmer database path")
    parser.add_argument("-rle", "--reported_lysin_EAD", required=True, type=str, help="reported lysin EAD structures(hmm files)")
    parser.add_argument("-bp", "--bacteriaORphage", required=True, type=str, help="bacteria pipeline or phage pipeline('B' for bacteria, 'P' for phage")
    parser.add_argument("-r", "--ref", default='', required=False, type=str, help="Reference lysins sequences")
    Args = parser.parse_args()
    
    if Args.bacteriaORphage == 'B':
        if Args.workdir[-1] == '/':
            resultdir = os.path.basename(Args.workdir[:-1])
        elif Args.workdir[-1] == "\\":
            resultdir = os.path.basename(Args.workdir[:-1])
        else:
            resultdir = os.path.basename(Args.workdir)
        
        if os.path.isdir(os.path.dirname(os.path.abspath(Args.workdir)) +'/' + resultdir + '/') == True:
            pass
        else:
            os.mkdir(os.path.dirname(os.path.abspath(Args.workdir)) +'/' + resultdir + '/')
        
        
        tl = tools()
        # step 1 prokka annotates ORFs
        curr_dir = sub.getoutput('pwd')
        os.chdir(Args.workdir)
        if os.path.isdir('./prokka_result/') == True:
            pass
        else:
            os.mkdir('./prokka_result/')

        target = Args.path
        curr_dir_target = curr_dir
        if target[-1] == '/':
            target = target
        elif target[-1] != '/':
            target = target + '/'

        if target[0] == '.':
            if target[1] == '/':
                target_suffix = target[1:]
            elif target[1] == '.':
                curr_dir_target = os.path.abspath(os.path.join(os.path.dirname(curr_dir + '/'), os.path.pardir))
                target_suffix = target[2:]
        else:
            target_suffix = target
            curr_dir_target = ''

        type_annotation = Args.type
        for i in os.listdir(curr_dir_target + target_suffix):
            lis = i.split('.')[:-1]
            name = '.'.join(lis)
            suffix = i.split('.')[-1]
            cmd_1 = tl.run_prokka(curr_dir_target + target_suffix + i,
                              './prokka_result/' + name + '/',name,type_annotation)
            tl.run(cmd_1)

        # step 2 phispy predict prophage
        if os.path.isdir('./phispy_out/') == True:
            pass
        else:
            os.mkdir('./phispy_out/')
        for i in os.listdir('./prokka_result/'):
            cmd_2 = tl.run_phispy('./prokka_result/' + i + '/' + i + '.gbk',
                              './phispy_out/' + i, i,0)
            tl.run(cmd_2)

        # step 3 select 1,2,3,4 colum for coordinate.tsv
        if os.path.isdir('./ppn/') == True:
            pass
        else:
            os.mkdir('./ppn/')

        fna_suffix = os.path.splitext(os.listdir(curr_dir_target + target_suffix)[0])[-1]
        for i in os.listdir('./phispy_out/'):
            if os.path.isdir('./ppn/' + i) == True:
                pass
            else:
                os.mkdir('./ppn/' + i)
            prophage_select('./phispy_out/'+ i + '/' + i + '_prophage_coordinates.tsv',
                            curr_dir_target + target_suffix + i + fna_suffix,'./ppn/' + i + '/' + i)


        # step 4 phanotate annotates prophage ORFs
        for i in os.listdir('./phispy_out/'):
            for j in os.listdir('./ppn/' + i):
                j_prefix = '.'.join(j.split('.')[:-1])
                j_suffix = j.split('.')[-1]
                if j_suffix == 'fasta':
                    cmd_3 = tl.run_phanotate('./ppn/' + i + '/' + j,
                                             './ppn/' + i + '/' + j_prefix + '.out')

                    tl.run(cmd_3)


        if os.path.isdir('./orf_ffn/') == True:
            pass
        else:
            os.mkdir('./orf_ffn/')
        for i in os.listdir('./phispy_out/'):
            for j in os.listdir('./ppn/' + i):
                j_prefix = '.'.join(j.split('.')[:-1])
                j_suffix = j.split('.')[-1]
                if j_suffix == 'fasta':
                    Gene_element_abstract('./ppn/' + i + '/' + j_prefix + '.out',
                                          './ppn/' + i + '/' + j,
                                          './orf_ffn/' + j_prefix + '.ffn')

        
        if len(os.listdir('./orf_ffn/')) == 0:
          # os.system('rm -r ./orf_ffn/ ./phispy_out/ ./ppn/ ./prokka_result/')
          raise('No prophages ORFs found!')
   
        else:
          # step 5 ppn faa together
          os.system('cat ./orf_ffn/* > all_protein_ut.faa')
          
          fa_dict = fasta2dict('./all_protein_ut.faa')
          filters = ["B","Z","J","O","U","X",'*']
  
          with open('./all_protein.faa','w') as f:
              for key in fa_dict:
                  if all(f not in fa_dict[key] for f in filters):
                      line = key + '\n' + fa_dict[key] + '\n'
                      f.write(line)
          f.close()
          
          os.system('cat %s > %s' % ('./all_protein.faa', './all_protein_tmp.txt'))
          with open('./all_protein.faa', 'w') as w:
            f = open('./all_protein_tmp.txt')
            for line in f:
              if line.startswith('>'):
                first_line = line[1::].strip()
                name = first_line.split(' ')[0]
                w.write('>' + name + '\n')
              else:
                w.write(line)
          w.close()
  
          if not os.path.getsize('./all_protein.faa'):
            with open('./putative_lysins.fa','w') as w:
              line = 'No lysins found!'
              w.write(line)
            w.close()
            
            # os.system('rm -r ./prokka_result/')
            # os.remove('./all_protein.faa')
            # os.remove('./all_protein_ut.faa')
          
          else:
            # step 6 cdhit cluster
            cmd_4 = tl.run_cdhit('./all_protein.faa','./all_protein_cdhit.faa',Args.cdhit_cutoff)
            tl.run(cmd_4)
    
            # step 7 calculate molecular weight
            molecular_weight('./all_protein_cdhit.faa','./all_protein_cdhit_filter.faa', float(Args.MWU),float(Args.MWL))
    
    
            # step 8 hmmsearch reported lysin structure in pfam
            if os.path.isdir('./hmmer_out/') == True:
                pass
            else:
                os.mkdir('./hmmer_out/')
    
            hmmer_db = Args.hmmer_db
            if hmmer_db[0] == '.':
                if hmmer_db[1] == '/':
                    hmmer_db_suffix = hmmer_db[1:]
                    curr_dir_hmmerdb = curr_dir
                elif hmmer_db[1] == '.':
                    curr_dir_hmmerdb = os.path.abspath(os.path.join(os.path.dirname(curr_dir + '/'), os.path.pardir))
                    hmmer_db_suffix = hmmer_db[2:]
            else:
                hmmer_db_suffix = hmmer_db
                curr_dir_hmmerdb = ''
    
            cmd_5 = tl.run_hmmsearch('./hmmer_out/all_protein_filter_hmmer_out.txt', Args.hmmer_cutoff,
                                     curr_dir_hmmerdb + hmmer_db_suffix,
                                     './all_protein_cdhit_filter.faa')
            tl.run(cmd_5)
            
            cmd_5_p = tl.run_hmmsearch_2('./hmmer_out/all_protein.txt', Args.hmmer_cutoff,
                                       curr_dir_hmmerdb + hmmer_db_suffix,
                                       './all_protein_cdhit_filter.faa')
            tl.run(cmd_5_p)
    
            reported_lysin = Args.reported_lysin
            if reported_lysin[0] == '.':
                if reported_lysin[1] == '/':
                    reported_lysin_suffix = reported_lysin[1:]
                    curr_dir_rp = curr_dir
                elif reported_lysin[1] == '.':
                    curr_dir_rp = os.path.abspath(os.path.join(os.path.dirname(curr_dir + '/'), os.path.pardir))
                    reported_lysin_suffix = reported_lysin[2:]
            else:
                reported_lysin_suffix = reported_lysin
                curr_dir_rp = ''
    
            find_pfam('./all_protein_cdhit_filter.faa', curr_dir_rp + reported_lysin_suffix)
            
            
            # step 9 Filter sequences without EAD
            if os.path.isdir('./hmmer_out_EAD/') == True:
                pass
            else:
                os.mkdir('./hmmer_out_EAD/')
    
            hmmer_db_EAD = Args.hmmer_db_EAD
            if hmmer_db_EAD[0] == '.':
                if hmmer_db_EAD[1] == '/':
                    hmmer_db_EAD_suffix = hmmer_db_EAD[1:]
                    curr_dir_hmmerdb_EAD = curr_dir
                elif hmmer_db_EAD[1] == '.':
                    curr_dir_hmmerdb_EAD = os.path.abspath(os.path.join(os.path.dirname(curr_dir + '/'), os.path.pardir))
                    hmmer_db_EAD_suffix = hmmer_db_EAD[2:]
            else:
                hmmer_db_EAD_suffix = hmmer_db_EAD
                curr_dir_hmmerdb_EAD = ''
    
            cmd_6 = tl.run_hmmsearch('./hmmer_out_EAD/all_protein_filter_hmmer_out_EAD.txt', Args.hmmer_cutoff,
                                     curr_dir_hmmerdb_EAD + hmmer_db_EAD_suffix,
                                     './all_protein_pfam_protein.fasta')
            tl.run(cmd_6)
    
            reported_lysin_EAD = Args.reported_lysin_EAD
            if reported_lysin_EAD[0] == '.':
                if reported_lysin_EAD[1] == '/':
                    reported_lysin_EAD_suffix = reported_lysin_EAD[1:]
                    curr_dir_rpe = curr_dir
                elif reported_lysin_EAD[1] == '.':
                    curr_dir_rpe = os.path.abspath(os.path.join(os.path.dirname(curr_dir + '/'), os.path.pardir))
                    reported_lysin_EAD_suffix = reported_lysin_EAD[2:]
            else:
                reported_lysin_EAD_suffix = reported_lysin_EAD
                curr_dir_rpe = ''
    
            find_pfam_EAD('./all_protein_pfam_protein.fasta', curr_dir_rpe + reported_lysin_EAD_suffix)
            
    
            # step 10 combine results of CAZY and pfam
            os.system('cat all_protein_pfam_protein_EAD.fasta > pfam_EAD.fasta')
            cmd_7 = tl.run_cdhit('./pfam_EAD.fasta', './pfam_EAD_cdhit.fasta', int(1))
            tl.run(cmd_7)
    
            # step 11 remove TMhelix
            tot = sub.getoutput("grep '>' %s | wc -l" % ('./pfam_EAD_cdhit.fasta'))
        
            if int(tot) > 100:
                num_1 = int(tot)//100
                num_2 = int(tot)%100
                Split_fa('./pfam_EAD_cdhit.fasta', tot, num_1, num_2)
              
                for i in range(1, int(num_1) + 1):
                   time_sleep = random.uniform(60, 180)
                   time.sleep(time_sleep)
                   cmd_8 = tl.run_deeptmhmm('./pfam_EAD_cdhit-' + str(i) + '00.fasta')
                   tl.run(cmd_8)
                   
                os.system('cat ./biolib_results/predicted_topologies.3line* > predicted_topologies.line')
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
              
              with open('./Domain_Info.txt', 'w') as w2:
                for line in f2:
                  if line[0] != "#" and len(line.split())!=0:
                    arr = line.strip().split(" ")
                    arr = list(filter(None, arr))
                    name = arr[0]
                    if name in dic_fa.keys():
                      li = arr[0] + '\t' + arr[3] + '\t' + arr[4].split('.')[0] + '\t' + arr[21] + '\t' + arr[19] + '-' + arr[20] + '\n'
                      print(li)
                      w2.write(li)
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
              
              
              if Args.ref != '':
                ref_lysins = str(Args.ref)
                first_dict = SeqIO.to_dict(SeqIO.parse(open('./putative_lysins.fa'),'fasta'))
                second_dict = SeqIO.to_dict(SeqIO.parse(open(ref_lysins),'fasta'))
                
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
                    line = key + '\t' + '\t'.join(dic_info[key][0:2]) + '\t' + ';'.join(dic_info[key][2:len(dic_info[key])-1]) + '\t' + dic_info[key][-1] + '\t' + '\t'.join(dic_ref[key]) + '\n'
                    w.write(line)
                w.close()
                
                        
              elif Args.ref == '':
                print('aaaaa')
                
                with open('./putative_lysins_info.txt','w') as w:
                  line = 'ID' + '\t' + 'MW' + '\t' + 'Length' + '\t' + 'Domains' + '\t' + 'Signalp' + '\n'
                  w.write(line)
                  for key in dic_info:
                    line = key + '\t' + '\t'.join(dic_info[key][0:2]) + '\t' + ';'.join(dic_info[key][2:len(dic_info[key])-1]) + '\t' + dic_info[key][-1] + '\n'
                    w.write(line)
                w.close()
                  
            else:
              print(state)
              
            # os.system('rm -r ./hmmer_out/ ./hmmer_out_EAD/ ./orf_ffn/ ./phispy_out/ ./ppn/ ./prokka_result/ ./biolib_results/')
            # os.system('rm -r ./pfam_EAD_cdhit*')
            # os.remove('./all_protein_cdhit.faa')
            # os.remove('./all_protein_cdhit.faa.clstr')
            # os.remove('./all_protein_cdhit_filter.faa')
            # os.remove('./all_protein.faa')
            # os.remove('./all_protein_pfam_protein.fasta')
            # os.remove('./all_protein_pfam_protein_EAD.fasta')
            # os.remove('./pfam_EAD.fasta')
            # os.remove('./all_protein_ut.faa')

    elif Args.bacteriaORphage == 'P':
        if Args.workdir[-1] == '/':
            resultdir = os.path.basename(Args.workdir[:-1])
        elif Args.workdir[-1] == "\\":
            resultdir = os.path.basename(Args.workdir[:-1])
        else:
            resultdir = os.path.basename(Args.workdir)
        
        if os.path.isdir(os.path.dirname(os.path.abspath(Args.workdir)) +'/' + resultdir + '/') == True:
            pass
        else:
            os.mkdir(os.path.dirname(os.path.abspath(Args.workdir)) +'/' + resultdir + '/')
              

        tl = tools()        
        # step 1 prokka annotates ORFs
        curr_dir = sub.getoutput('pwd')
        os.chdir(Args.workdir)
        if os.path.isdir('./prokka_result/') == True:
            pass
        else:
            os.mkdir('./prokka_result/')

        target = Args.path
        curr_dir_target = curr_dir
        if target[-1] == '/':
            target = target
        elif target[-1] != '/':
            target = target + '/'

        if target[0] == '.':
            if target[1] == '/':
                target_suffix = target[1:]
            elif target[1] == '.':
                curr_dir_target = os.path.abspath(os.path.join(os.path.dirname(curr_dir + '/'), os.path.pardir))
                target_suffix = target[2:]
        else:
            target_suffix = target
            curr_dir_target = ''

        type_annotation = Args.type
        for i in os.listdir(curr_dir_target + target_suffix):
            lis = i.split('.')[:-1]
            name = '.'.join(lis)
            suffix = i.split('.')[-1]
            cmd_1 = tl.run_prokka(curr_dir_target + target_suffix + i,
                              './prokka_result/' + name + '/',name,type_annotation)
            tl.run(cmd_1)
            
        for i in os.listdir('./prokka_result/'):
          for j in os.listdir('./prokka_result/' + i):
            if j.endswith('.faa'):
              os.system('cat %s > %s' % ('./prokka_result/' + i + '/' + j, './prokka_result/' + i + '/tmp.txt'))
              with open('./prokka_result/' + i + '/' + j, 'w') as w:
                f = open('./prokka_result/' + i + '/tmp.txt')
                for line in f:
                  if line.startswith('>'):
                    print(line)
                    first_line = line[1::].strip()
                    key = first_line.split(' ')[0].split('_')[0]
                    num = first_line.split(' ')[0].split('_')[1]
                    lis = j.split('.')[:-1]
                    name = '.'.join(lis)
                    w.write('>' + name + '_' + num + '\n')
                  else:
                    w.write(line)
              w.close()

        

        # step 2 move faa into phage_faa fold
        if os.path.isdir('./phage_faa/') == True:
            pass
        else:
            os.mkdir('./phage_faa/')
            
        for i in os.listdir('./prokka_result/'):
            for j in os.listdir('./prokka_result/' + i):
                if os.path.splitext(j)[-1] == ".faa":
                    os.system('cp %s %s' % ('./prokka_result/' + i + '/' + j, './phage_faa/'))

        
        if len(os.listdir('./phage_faa/')) == 0:
          # os.system('rm -r ./prokka_result/ ./phage_faa/')
          raise('No phage faa found!')
          
        else:
          # step 3 phage faa together
          os.system('cat ./phage_faa/* > all_protein_ut.faa')
          
          fa_dict = fasta2dict('./all_protein_ut.faa')
  
          filters = ["B","Z","J","O","U","X",'*']
          with open('./all_protein.faa','w') as f:
              for key in fa_dict:
                  if all(f not in fa_dict[key] for f in filters):
                      line = key + '\n' + fa_dict[key] + '\n'
                      f.write(line)
          f.close()
          
          if not os.path.getsize('./all_protein.faa'):
            with open('./putative_lysins.fa','w') as w:
              line = 'No lysins found!'
              w.write(line)
            w.close()
            
            # os.system('rm -r ./orf_ffn/ ./phispy_out/ ./ppn/ ./prokka_result/')
            # os.remove('./all_protein.faa')
            # os.remove('./all_protein_ut.faa')
          
          else:
            # step 4 cdhit cluster
            cmd_4 = tl.run_cdhit('./all_protein.faa','./all_protein_cdhit.faa',Args.cdhit_cutoff)
            tl.run(cmd_4)
    
            # step 5 calculate molecular weight
            molecular_weight('./all_protein_cdhit.faa','./all_protein_cdhit_filter.faa', float(Args.MWU),float(Args.MWL))
    
    
            # step 6 hmmsearch reported lysin structure in pfam
            if os.path.isdir('./hmmer_out/') == True:
                pass
            else:
                os.mkdir('./hmmer_out/')
    
            hmmer_db = Args.hmmer_db
            if hmmer_db[0] == '.':
                if hmmer_db[1] == '/':
                    hmmer_db_suffix = hmmer_db[1:]
                    curr_dir_hmmerdb = curr_dir
                elif hmmer_db[1] == '.':
                    curr_dir_hmmerdb = os.path.abspath(os.path.join(os.path.dirname(curr_dir + '/'), os.path.pardir))
                    hmmer_db_suffix = hmmer_db[2:]
            else:
                hmmer_db_suffix = hmmer_db
                curr_dir_hmmerdb = ''
    
            cmd_5 = tl.run_hmmsearch('./hmmer_out/all_protein_filter_hmmer_out.txt', Args.hmmer_cutoff,
                                     curr_dir_hmmerdb + hmmer_db_suffix,
                                     './all_protein_cdhit_filter.faa')
            tl.run(cmd_5)
            
            cmd_5_p = tl.run_hmmsearch_2('./hmmer_out/all_protein.txt', Args.hmmer_cutoff,
                                       curr_dir_hmmerdb + hmmer_db_suffix,
                                       './all_protein_cdhit_filter.faa')
            tl.run(cmd_5_p)
    
            reported_lysin = Args.reported_lysin
            if reported_lysin[0] == '.':
                if reported_lysin[1] == '/':
                    reported_lysin_suffix = reported_lysin[1:]
                    curr_dir_rp = curr_dir
                elif reported_lysin[1] == '.':
                    curr_dir_rp = os.path.abspath(os.path.join(os.path.dirname(curr_dir + '/'), os.path.pardir))
                    reported_lysin_suffix = reported_lysin[2:]
            else:
                reported_lysin_suffix = reported_lysin
                curr_dir_rp = ''
    
            find_pfam('./all_protein_cdhit_filter.faa', curr_dir_rp + reported_lysin_suffix)
            
            
            # step 7 Filter sequences without EAD
            if os.path.isdir('./hmmer_out_EAD/') == True:
                pass
            else:
                os.mkdir('./hmmer_out_EAD/')
    
            hmmer_db_EAD = Args.hmmer_db_EAD
            if hmmer_db_EAD[0] == '.':
                if hmmer_db_EAD[1] == '/':
                    hmmer_db_EAD_suffix = hmmer_db_EAD[1:]
                    curr_dir_hmmerdb_EAD = curr_dir
                elif hmmer_db_EAD[1] == '.':
                    curr_dir_hmmerdb_EAD = os.path.abspath(os.path.join(os.path.dirname(curr_dir + '/'), os.path.pardir))
                    hmmer_db_EAD_suffix = hmmer_db_EAD[2:]
            else:
                hmmer_db_EAD_suffix = hmmer_db_EAD
                curr_dir_hmmerdb_EAD = ''
    
            cmd_6 = tl.run_hmmsearch('./hmmer_out_EAD/all_protein_filter_hmmer_out_EAD.txt', Args.hmmer_cutoff,
                                     curr_dir_hmmerdb_EAD + hmmer_db_EAD_suffix,
                                     './all_protein_pfam_protein.fasta')
            tl.run(cmd_6)
    
            reported_lysin_EAD = Args.reported_lysin_EAD
            if reported_lysin_EAD[0] == '.':
                if reported_lysin_EAD[1] == '/':
                    reported_lysin_EAD_suffix = reported_lysin_EAD[1:]
                    curr_dir_rpe = curr_dir
                elif reported_lysin_EAD[1] == '.':
                    curr_dir_rpe = os.path.abspath(os.path.join(os.path.dirname(curr_dir + '/'), os.path.pardir))
                    reported_lysin_EAD_suffix = reported_lysin_EAD[2:]
            else:
                reported_lysin_EAD_suffix = reported_lysin_EAD
                curr_dir_rpe = ''
    
            find_pfam_EAD('./all_protein_pfam_protein.fasta', curr_dir_rpe + reported_lysin_EAD_suffix)
            
    
            # step 9 combine results of CAZY and pfam
            os.system('cat all_protein_pfam_protein_EAD.fasta > pfam_EAD.fasta')
            cmd_7 = tl.run_cdhit('./pfam_EAD.fasta', './pfam_EAD_cdhit.fasta', int(1))
            tl.run(cmd_7)
    
            # step 12 remove TMhelix
            tot = sub.getoutput("grep '>' %s | wc -l" % ('./pfam_EAD_cdhit.fasta'))
        
            if int(tot) > 100:
                num_1 = int(tot)//100
                num_2 = int(tot)%100
                Split_fa('./pfam_EAD_cdhit.fasta', tot, num_1, num_2)
              
                for i in range(1, int(num_1) + 1):
                   time_sleep = random.uniform(60, 180)
                   time.sleep(time_sleep)
                   cmd_8 = tl.run_deeptmhmm('./pfam_EAD_cdhit-' + str(i) + '00.fasta')
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
              
              
              with open('./Domain_Info.txt', 'w') as w2:
                for line in f2:
                  if line[0] != "#" and len(line.split())!=0:
                    arr = line.strip().split(" ")
                    arr = list(filter(None, arr))
                    name = arr[0]
                    if name in dic_fa.keys():
                      li = arr[0] + '\t' + arr[3] + '\t' + arr[4].split('.')[0] + '\t' + arr[21] + '\t' + arr[19] + '-' + arr[20] + '\n'
                      print(li)
                      w2.write(li)
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
              
              
              if Args.ref != '':
                ref_lysins = str(Args.ref)
                first_dict = SeqIO.to_dict(SeqIO.parse(open('./putative_lysins.fa'),'fasta'))
                second_dict = SeqIO.to_dict(SeqIO.parse(open(ref_lysins),'fasta'))
                
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
                    line = key + '\t' + '\t'.join(dic_info[key][0:2]) + '\t' + ';'.join(dic_info[key][2:len(dic_info[key])-1]) + '\t' + dic_info[key][-1] + '\t' + '\t'.join(dic_ref[key]) + '\n'
                    w.write(line)
                w.close()
                
                        
              elif Args.ref == '':
                with open('./putative_lysins_info.txt','w') as w:
                  line = 'ID' + '\t' + 'MW' + '\t' + 'Length' + '\t' + 'Domains' + '\t' + 'Signalp' + '\n'
                  w.write(line)
                  for key in dic_info:
                    line = key + '\t' + '\t'.join(dic_info[key][0:2]) + '\t' + ';'.join(dic_info[key][2:len(dic_info[key])-1]) + '\t' + dic_info[key][-1] + '\n'
                    w.write(line)
                w.close()
                
                    
            else:
              print(state)
              
              
            # os.system('rm -r ./hmmer_out/ ./hmmer_out_EAD/ ./prokka_result/ ./biolib_results/')
            # os.system('rm -r ./pfam_EAD_cdhit*')
            # os.remove('./all_protein_cdhit.faa')
            # os.remove('./all_protein_cdhit.faa.clstr')
            # os.remove('./all_protein_cdhit_filter.faa')
            # os.remove('./all_protein.faa')
            # os.remove('./all_protein_pfam_protein.fasta')
            # os.remove('./all_protein_pfam_protein_EAD.fasta')
            # os.remove('./pfam_EAD.fasta')
            # os.remove('./all_protein_ut.faa')

    else:
        raise('Error, please check parameter "--bp"')
