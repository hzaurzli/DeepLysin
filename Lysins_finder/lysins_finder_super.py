#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Lysins_finder_super.py 
#
#  Copyright 2025 Small runze
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


def add_length(ead_path, cbd_path, blast_res, blast_len_res):
    f_ead = open(ead_path)
    f_cbd = open(cbd_path)
    
    dict_ead = {}
    dict_cbd = {}
    dict_domain = {}
    
    for m in f_ead:
      lis = []
      id = 'CDD:' + m.strip().split(',')[2]
      domain_id = m.strip().split(',')[0]
      domain_name = m.strip().split(',')[1]
      length = m.strip().split(',')[3]
      lis.append(domain_id)
      lis.append(domain_name)
      lis.append(length)
      dict_ead[id] = lis
      dict_domain[id] = lis
      
    for n in f_cbd:
      lis = []
      id = 'CDD:' + n.strip().split(',')[2]
      domain_id = n.strip().split(',')[0]
      domain_name = n.strip().split(',')[1]
      length = n.strip().split(',')[3]
      lis.append(domain_id)
      lis.append(domain_name)
      lis.append(length)
      dict_cbd[id] = lis
      dict_domain[id] = lis
    
    for i in os.listdir(blast_res):
      with open(blast_len_res + '/' + i, 'w') as w:
        f = open(blast_res + '/' + i)
        for j in f:
          item = j.strip().split('\t')
          if item[1] in dict_ead:
            add_length = item[1] + '::' + dict_ead[item[1]][2]
            item[1] = add_length
          if item[1] in dict_cbd:
            add_length = item[1] + '::' + dict_cbd[item[1]][2]
            item[1] = add_length
            
          line = '\t'.join(item) + '\n'
          w.write(line)
      w.close()
      
    return dict_ead, dict_cbd, dict_domain
      

def blast_filter(out_folder, coverage):
    Domain_location_dict = {}
    position_info = {}
    isolates_list = []
    rootdir1 = out_folder
    for i in os.listdir(rootdir1):
        file = i
        blast_info = open(out_folder + '/' + i, "r")
        id = '.'.join(file.split('.')[:-1])
        isolates_list.append(id)
        for line in blast_info:
            line_info = line.strip().split("\t")
            Contig_ID_info = line_info[0]
            Domain_ID = line_info[1].split("::")[0]
            Domain_len = line_info[1].split("::")[-1]
            identical_percent = line_info[2]
            align_length = int(line_info[3])
            contig_start = int(line_info[6])
            contig_end = int(line_info[7])
            Domain_start = int(line_info[8])
            Domain_end = int(line_info[9])
            Score = float(line_info[11])
            F_R = "F"
            if Domain_end < Domain_start:
                F_R = "R"
            align_percent = '%.2f' % (float(align_length) / float(Domain_len) * 100)
            key_use = (id)
            
            position_lis = []
            position_element = Domain_ID + '&' + str(contig_start) + '-' + str(contig_end) + ':' + F_R + '&' + str(identical_percent)
            if Contig_ID_info in position_info:
              position_info[Contig_ID_info].append(position_element)
            else:
              position_lis.append(position_element)
              position_info[Contig_ID_info] = position_lis
            
            if float(align_percent) > coverage:
                Domain_location_dict.setdefault(key_use, []).append(
                    (Domain_ID, contig_start,
                     contig_end, Score,
                     identical_percent, align_percent,
                     Contig_ID_info, F_R, Domain_len))
        blast_info.close()
    return Domain_location_dict, isolates_list, position_info
    

def Domain_filter(Domain_location_dict, isolates_list, ident, coverage, over_lap):
    Domain_location_filter_dict = {}
    Domain_list_get = []
    for item in Domain_location_dict.items():
        key_data = item[0]
        Domain_list = item[1]
        Domain_list.sort(key=operator.itemgetter(1))
        start_initial = 0
        ii_keep = 0
        Domain_filter_list = []
        for ii in range(len(Domain_list)):
            if Domain_list[ii][1] >= start_initial:
                Domain_filter_list.append((Domain_list[ii][0], Domain_list[ii][4], Domain_list[ii][5], Domain_list[ii][1],
                                        Domain_list[ii][2], Domain_list[ii][6], Domain_list[ii][7]))
                start_initial = Domain_list[ii][2]
                ii_keep = ii
            elif Domain_list[ii][1] < start_initial and Domain_list[ii][2] > start_initial and float(
                Domain_list[ii][2] - start_initial) / float(
                Domain_list[ii][2] - Domain_list[ii][1]) > over_lap / 100 and float(
                start_initial - Domain_list[ii][1]) / float(Domain_list[ii][8]) < (100 - over_lap) / 100:
                Domain_filter_list.append((Domain_list[ii][0], Domain_list[ii][4], Domain_list[ii][5], Domain_list[ii][1],
                                        Domain_list[ii][2], Domain_list[ii][6], Domain_list[ii][7]))
                start_initial = Domain_list[ii][2]
                ii_keep = ii
            else:
                if Domain_list[ii][3] > Domain_list[ii_keep][3]:
                    sss = (Domain_list[ii_keep][0], Domain_list[ii_keep][4], Domain_list[ii_keep][5], Domain_list[ii][1],
                           Domain_list[ii][2], Domain_list[ii][6], Domain_list[ii][7])
                    if sss in Domain_filter_list:
                        Domain_filter_list.remove(sss)
                    Domain_filter_list.append((Domain_list[ii][0], Domain_list[ii][4], Domain_list[ii][5], Domain_list[ii][1],
                                            Domain_list[ii][2], Domain_list[ii][6], Domain_list[ii][7]))
                    start_initial = Domain_list[ii][2]
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
            if float(i[1]) >= ident and float(i[2]) >= coverage:  
                Domain_location_use_dict.setdefault(ID_filter, []).append(i)
    for jj in isolates_list:
        if jj not in Domain_location_use_dict:
            Domain_location_use_dict[jj] = "No Lysins"
    return Domain_location_use_dict, Domain_list_get


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
    parser.add_argument("-t", "--type", required=False, default='', type=str, help="prokka kingdom type")
    parser.add_argument("-pp", "--prophage_method", required=False, default='phispy', type=str, help="prophage predict method")
    parser.add_argument("-ds", "--dbscan_swa", required=False, type=str, default='', help="path of dbscan-swa.py")
    parser.add_argument("-c", "--cdhit_cutoff", default=0.95,required=False, type=float, help="cdhit cluster cutoff")
    parser.add_argument("-wkdir", "--workdir", required=True, type=str, help="work directory")
    parser.add_argument("-mu", "--MWU", required=False, default=50000, type=float, help="upper proteins molecular weight")
    parser.add_argument("-ml", "--MWL", required=False, default=10000, type=float, help="lower proteins molecular weight")
    parser.add_argument("-r", "--ref", default='', required=False, type=str, help="reference lysins sequences")
    parser.add_argument("-m", "--method", default='hmmer', required=True, type=str, help="searching method 'hmmer' or 'rpsblast'")
    parser.add_argument("-bp", "--bacteriaORphage", required=True, type=str, help="bacteria pipeline or phage pipeline('B' for bacteria, 'P' for phage")
    
    parser.add_argument("-hc", "--hmmer_cutoff", default=1e-5,required=False, type=float, help="hmmer search evalue cutoff(hmmer)")
    parser.add_argument("-hdc", "--hmmer_db_CBD", required=False, type=str, help="CBD hmmer database path(hmmer)")
    parser.add_argument("-rlc", "--reported_lysin_CBD", required=False, type=str, help="reported lysin CBD structures(hmm files,hmmer)")
    parser.add_argument("-hde", "--hmmer_db_EAD", required=False, type=str, help="EAD hmmer database path(hmmer)")
    parser.add_argument("-rle", "--reported_lysin_EAD", required=False, type=str, help="reported lysin EAD structures(hmm files,hmmer)")
    
    parser.add_argument("-EI", "--EAD_info", required=False, type=str, help="EAD information(rpsblast)")
    parser.add_argument("-CI", "--CBD_info", required=False, type=str, help="CBD information(rpsblast)")
    parser.add_argument("-rc", "--rpsblast_cutoff", default=1e-5,required=False, type=float, help="rpsblast evalue cutoff(rpsblast)")
    parser.add_argument("-rcov", "--rpsblast_coverage", required=False, default=80, type=float, help="rpbblast region coverage(rpsblast)")
    parser.add_argument("-rident", "--rpsblast_identity", required=False, default=20, type=float, help="rpbblast identity(rpsblast)")
    parser.add_argument("-rol", "--rpsblast_over_lap", default=80,required=False, type=float, help="rpsblast cutoff of overlap in the same region(rpsblast)")
    parser.add_argument("-rpsdb", "--rpsblast_db", required=False, type=str, help="reported lysin rpsblast database path(need db index, rpsblast)")
    Args = parser.parse_args()
    
    if Args.method == 'hmmer':
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
            curr_dir = sub.getoutput('pwd')
            os.chdir(Args.workdir)
            
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
            
            
            if Args.prophage_method == 'DBSCAN_SWA':
              if Args.type == '':
                  if Args.dbscan_swa != '':
                    # step 2 DBSCAN-SWA predict prophage
                    if os.path.isdir('./DBSCAN_SWA_out/') == True:
                        pass
                    else:
                        os.mkdir('./DBSCAN_SWA_out/')
                                   
                    for i in os.listdir(curr_dir_target + target_suffix):
                        lis = i.split('.')[:-1]
                        prefix = '.'.join(lis)
                        cmd_2 = tl.run_DBSCAN_SWA(Args.dbscan_swa, curr_dir_target + target_suffix + i,
                                                  './DBSCAN_SWA_out/' + prefix, prefix)
                        tl.run(cmd_2)
                     
                        
                    if os.path.isdir('./orf_ffn/') == True:
                        pass
                    else:
                        os.mkdir('./orf_ffn/')
                        
                    # step 4 phanotate annotates prophage ORFs
                    for i in os.listdir('./DBSCAN_SWA_out/'):
                        for j in os.listdir('./DBSCAN_SWA_out/' + i):
                            j_prefix = '.'.join(j.split('.')[:-1])
                            j_suffix = j.split('.')[-1]
                            if j_suffix == 'faa':
                                os.system('mv %s %s' % ('./DBSCAN_SWA_out/' + i + '/' + j_prefix + '.' + j_suffix,
                                                        './DBSCAN_SWA_out/' + i + '/' + j_prefix + '_tmp.' + j_suffix))
                                f = open('./DBSCAN_SWA_out/' + i + '/' + j_prefix + '_tmp.' + j_suffix)
                                with open('./DBSCAN_SWA_out/' + i + '/' + j_prefix + '.' + j_suffix, 'w') as w:
                                    for m in f:
                                        if m != None:
                                            if m.startswith('>'):
                                                line = '>' + i + ':' + m.strip().split('|')[3].split('_')[0] + '-' + m.strip().split('|')[3].split('_')[1] + ':' + m.strip().split('|')[3].split('_')[2] + ':' + m.strip().split('|')[2] + '\n'
                                                w.write(line)
                                            else:
                                                line = m
                                                w.write(line)
                                w.close()
                                os.remove('./DBSCAN_SWA_out/' + i + '/' + j_prefix + '_tmp.' + j_suffix)
                                    
                                shutil.copyfile('./DBSCAN_SWA_out/' + i + '/' + j, './orf_ffn/' + j)
                
                  else:
                      raise ValueError('please add dbscan-swa.py path!')     
                
              else:
                  raise ValueError('redundant parameters "-t" !')     
              
            elif Args.prophage_method == 'phispy':
              if Args.type == 'Bacteria':
                # step 1 prokka annotates ORFs
                if os.path.isdir('./prokka_result/') == True:
                    pass
                else:
                    os.mkdir('./prokka_result/')
                
                name_fna = {}
                type_annotation = Args.type
                for i in os.listdir(curr_dir_target + target_suffix):
                    lis = i.split('.')[:-1]
                    name = '.'.join(lis)
                    suffix = i.split('.')[-1]
                    name_fna[name] = suffix
                    cmd_1 = tl.run_prokka(curr_dir_target + target_suffix + i,
                                      './prokka_result/' + name + '/', name, type_annotation)
                    tl.run(cmd_1)
                    
              else:
                  raise ValueError('please correct annotation type!')
                  
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
      
      
              for i in os.listdir('./phispy_out/'):
                if os.path.isdir('./ppn/' + i) == True:
                  pass
                else:
                  os.mkdir('./ppn/' + i)
      
                fna_suffix = name_fna[i]
                prophage_select('./phispy_out/'+ i + '/' + i + '_prophage_coordinates.tsv',
                                curr_dir_target + target_suffix + i + '.' + fna_suffix,'./ppn/' + i + '/' + i)
      
      
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
                                                
                                                
              for name in os.listdir('./orf_ffn/'):
                  f = open('./orf_ffn/' + name)
                  prefix = '.'.join(name.split('.')[:-1])
                  os.system('mv %s %s' % ('./orf_ffn/' + prefix + '.ffn', './orf_ffn/' + prefix + '_tmp.ffn'))
                
                  with open('./orf_ffn/' + prefix + '.ffn', 'w') as w:
                    for jj in f:
                      if jj != None:
                        if jj.startswith('>'):
                          line = '>' + prefix + ':' + jj.strip().split(':')[2].split(' ')[0] + ':' + jj.strip().split(':')[1] + '\n'
                          w.write(line)
                        else:
                          line = jj
                          w.write(line)
                  w.close()
                  os.remove('./orf_ffn/' + prefix + '_tmp.ffn')
    
            datasize = []
            for i in os.listdir('./orf_ffn/'):
              stats = os.stat('./orf_ffn/' + i)
              datasize.append(stats.st_size)
              
              
            if all(element==0 for element in datasize) != 0:
              if os.path.isdir('./phispy_out/') == True:
                  os.system('rm -r ./orf_ffn/ ./phispy_out/ ./ppn/ ./prokka_result/')
                  raise ValueError('No prophages ORFs found!')
              else:
                  os.system('rm -r ./orf_ffn/ ./DBSCAN_SWA_out/')
                  raise ValueError('No prophages ORFs found!')
              
       
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
                
                if os.path.isdir('./phispy_out/') == True:
                    os.system('rm -r ./orf_ffn/ ./phispy_out/ ./ppn/ ./prokka_result/')
                else:
                    os.system('rm -r ./orf_ffn/ ./DBSCAN_SWA_out/')
                
                os.remove('./all_protein.faa')
                os.remove('./all_protein_ut.faa')
                os.remove('./all_protein_tmp.txt')
                
              
              else:
                # step 7 calculate molecular weight
                molecular_weight('./all_protein.faa','./all_protein_filter.faa', float(Args.MWU),float(Args.MWL))
        
        
                # step 8 hmmsearch reported lysin structure in pfam
                if os.path.isdir('./hmmer_out/') == True:
                    pass
                else:
                    os.mkdir('./hmmer_out/')
                
                if os.path.isdir('./hmmer_database/') == True:
                    pass
                else:
                    os.mkdir('./hmmer_database/')
        
                hmmer_db_CBD = Args.hmmer_db_CBD
                hmmer_db_EAD = Args.hmmer_db_EAD
                reported_lysin_CBD = Args.reported_lysin_CBD
                reported_lysin_EAD = Args.reported_lysin_EAD
                
                
                if hmmer_db_CBD[0] == '.':
                    if hmmer_db_CBD[1] == '/':
                        hmmer_db_CBD_suffix = hmmer_db_CBD[1:]
                        curr_dir_hmmerdb_CBD = curr_dir
                    elif hmmer_db_CBD[1] == '.':
                        curr_dir_hmmerdb_CBD = os.path.abspath(os.path.join(os.path.dirname(curr_dir + '/'), os.path.pardir))
                        hmmer_db_CBD_suffix = hmmer_db_CBD[2:]
                else:
                    hmmer_db_CBD_suffix = hmmer_db_CBD
                    curr_dir_hmmerdb_CBD = ''
                    
                    
                if os.path.isdir('./hmmer_out_EAD/') == True:
                    pass
                else:
                    os.mkdir('./hmmer_out_EAD/')
                
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
                    
                    
                if reported_lysin_CBD[0] == '.':
                    if reported_lysin_CBD[1] == '/':
                        reported_lysin_CBD_suffix = reported_lysin_CBD[1:]
                        curr_dir_rpc = curr_dir
                    elif reported_lysin_CBD[1] == '.':
                        curr_dir_rpc = os.path.abspath(os.path.join(os.path.dirname(curr_dir + '/'), os.path.pardir))
                        reported_lysin_CBD_suffix = reported_lysin_CBD[2:]
                else:
                    reported_lysin_CBD_suffix = reported_lysin_CBD
                    curr_dir_rpc = ''
                    
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
                
                
                os.system('cat %s %s > %s' % (curr_dir_hmmerdb_CBD + hmmer_db_CBD_suffix, 
                                              curr_dir_hmmerdb_EAD + hmmer_db_EAD_suffix,
                                              './hmmer_database/lysin_reported.hmm'))
                                              
                os.system('cat %s %s > %s' % (curr_dir_rpc + reported_lysin_CBD_suffix, 
                                              curr_dir_rpe + reported_lysin_EAD_suffix,
                                              './hmmer_database/lysin_reported.txt'))
                                              
                cmd_5 = tl.run_hmmsearch('./hmmer_out/all_protein_filter_hmmer_out.txt', Args.hmmer_cutoff,
                                         './hmmer_database/lysin_reported.hmm',
                                         './all_protein_filter.faa')
                tl.run(cmd_5)
                
                cmd_5_p = tl.run_hmmsearch_2('./hmmer_out/all_protein.txt', Args.hmmer_cutoff,
                                             './hmmer_database/lysin_reported.hmm',
                                             './all_protein_filter.faa')
                tl.run(cmd_5_p)
                find_pfam('./all_protein_filter.faa',  './hmmer_database/lysin_reported.txt')
                
                
                # step 9 Filter sequences without EAD
                cmd_6 = tl.run_hmmsearch('./hmmer_out_EAD/all_protein_filter_hmmer_out_EAD.txt', Args.hmmer_cutoff,
                                         curr_dir_hmmerdb_EAD + hmmer_db_EAD_suffix,
                                         './all_protein_pfam_protein.fasta')
                tl.run(cmd_6)
    
                if not os.path.getsize("./all_protein_pfam_protein.fasta"):
                    raise ValueError('No domain was found and No lysins found!!!')
        
                find_pfam_EAD('./all_protein_pfam_protein.fasta', curr_dir_rpe + reported_lysin_EAD_suffix)
                
        
                # step 10 combine results of CAZY and pfam
                os.system('cat all_protein_pfam_protein_EAD.fasta > pfam_EAD.fasta')
                
                dic_fa_ead = fasta2dict('./pfam_EAD.fasta')
                with open('./pfam_EAD_tmp.fasta','w') as f:
                  for key in dic_fa_ead:
                      if all(f not in dic_fa_ead[key] for f in filters):
                          line = key + '\n' + dic_fa_ead[key] + '\n'
                          f.write(line)
                f.close()
                
                
                x = float(Args.cdhit_cutoff)
                cdhit_cutoff = str("%.2f" %x)
                if cdhit_cutoff == '1.00':
                    os.system('cat ./pfam_EAD_tmp.fasta > ./pfam_EAD_cdhit.fasta')
                    print('cdhit_cutoff 1.00')
                else:
                    cmd_7 = tl.run_cdhit('./pfam_EAD_tmp.fasta', './pfam_EAD_cdhit.fasta', Args.cdhit_cutoff)
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
                       
                    os.system('cat ./biolib_results/predicted_topologies.3line* > ./biolib_results/predicted_topologies.line')
                    remove_TMhelix('./biolib_results/predicted_topologies.line','./pfam_EAD_cdhit.fasta','./putative_lysins.fa')
                  
                else:
                    cmd_8 = tl.run_deeptmhmm('./pfam_EAD_cdhit.fasta')
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
                  
                  domain_type = {}
                  f_CBD = open(curr_dir_rpc + reported_lysin_CBD_suffix)
                  for i in f_CBD:
                    item = i.strip()
                    domain_type[item] = 'CBD'
                  
                  f_EAD = open(curr_dir_rpe + reported_lysin_EAD_suffix)
                  for i in f_EAD:
                    item = i.strip()
                    domain_type[item] = 'EAD'
                  
                  print(domain_type)
                  
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
                  
                  
                  if Args.ref != '':
                    first_dict = SeqIO.to_dict(SeqIO.parse(open('./putative_lysins.fa'),'fasta'))
                    os.chdir(curr_dir)
                    ref_lysins = os.path.abspath(str(Args.ref))
                    second_dict = SeqIO.to_dict(SeqIO.parse(open(ref_lysins),'fasta'))
                    os.chdir(Args.workdir)
                    
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
                
                
                time.sleep(120) 
                if os.path.isdir('./phispy_out/') == True:
                    os.system('rm -r ./hmmer_database/ ./hmmer_out/ ./hmmer_out_EAD/ ./orf_ffn/ ./phispy_out/ ./ppn/ ./prokka_result/ ./biolib_results/')
                else:
                    os.system('rm -r ./hmmer_database/ ./hmmer_out/ ./hmmer_out_EAD/ ./orf_ffn/ ./DBSCAN_SWA_out/ ./biolib_results/') 
                os.system('rm -r ./pfam_EAD_cdhit*')
                os.remove('./all_protein_filter.faa')
                os.remove('./all_protein.faa')
                os.remove('./all_protein_pfam_protein.fasta')
                os.remove('./all_protein_pfam_protein_EAD.fasta')
                os.remove('./pfam_EAD.fasta')
                os.remove('./pfam_EAD_tmp.fasta')
                os.remove('./all_protein_tmp.txt')
                os.remove('./all_protein_ut.faa')
                os.remove('./molecular_weight.txt')
                os.remove('./MW_Length.txt') 
                os.remove('./Domain_Info.txt')
                os.system('rm -r ./signaltmp/')
    
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
            
            if Args.type == 'Viruses':
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
                      
            else:
                raise ValueError('please correct annotation type!')

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
              os.system('rm -r ./prokka_result/ ./phage_faa/')
              raise ValueError('No phage faa found!')
              
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
                
                os.system('rm -r ./prokka_result/')
                os.remove('./all_protein.faa')
                os.remove('./all_protein_ut.faa')
              
              else:
                # step 5 calculate molecular weight
                molecular_weight('./all_protein.faa','./all_protein_filter.faa', float(Args.MWU),float(Args.MWL))
        
                # step 6 hmmsearch reported lysin structure in pfam
                if os.path.isdir('./hmmer_out/') == True:
                    pass
                else:
                    os.mkdir('./hmmer_out/')
                
                if os.path.isdir('./hmmer_database/') == True:
                    pass
                else:
                    os.mkdir('./hmmer_database/')
        
                hmmer_db_CBD = Args.hmmer_db_CBD
                hmmer_db_EAD = Args.hmmer_db_EAD
                reported_lysin_CBD = Args.reported_lysin_CBD
                reported_lysin_EAD = Args.reported_lysin_EAD
                
                
                if hmmer_db_CBD[0] == '.':
                    if hmmer_db_CBD[1] == '/':
                        hmmer_db_CBD_suffix = hmmer_db_CBD[1:]
                        curr_dir_hmmerdb_CBD = curr_dir
                    elif hmmer_db_CBD[1] == '.':
                        curr_dir_hmmerdb_CBD = os.path.abspath(os.path.join(os.path.dirname(curr_dir + '/'), os.path.pardir))
                        hmmer_db_CBD_suffix = hmmer_db_CBD[2:]
                else:
                    hmmer_db_CBD_suffix = hmmer_db_CBD
                    curr_dir_hmmerdb_CBD = ''
                    
                    
                if os.path.isdir('./hmmer_out_EAD/') == True:
                    pass
                else:
                    os.mkdir('./hmmer_out_EAD/')
                
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
                    
                if reported_lysin_CBD[0] == '.':
                    if reported_lysin_CBD[1] == '/':
                        reported_lysin_CBD_suffix = reported_lysin_CBD[1:]
                        curr_dir_rpc = curr_dir
                    elif reported_lysin_CBD[1] == '.':
                        curr_dir_rpc = os.path.abspath(os.path.join(os.path.dirname(curr_dir + '/'), os.path.pardir))
                        reported_lysin_CBD_suffix = reported_lysin_CBD[2:]
                else:
                    reported_lysin_CBD_suffix = reported_lysin_CBD
                    curr_dir_rpc = ''
                    
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
                
                
                os.system('cat %s %s > %s' % (curr_dir_hmmerdb_CBD + hmmer_db_CBD_suffix, 
                                              curr_dir_hmmerdb_EAD + hmmer_db_EAD_suffix,
                                              './hmmer_database/lysin_reported.hmm'))
                                              
                os.system('cat %s %s > %s' % (curr_dir_rpc + reported_lysin_CBD_suffix, 
                                              curr_dir_rpe + reported_lysin_EAD_suffix,
                                              './hmmer_database/lysin_reported.txt'))
                                              
                cmd_5 = tl.run_hmmsearch('./hmmer_out/all_protein_filter_hmmer_out.txt', Args.hmmer_cutoff,
                                         './hmmer_database/lysin_reported.hmm',
                                         './all_protein_filter.faa')
                tl.run(cmd_5)
                
                cmd_5_p = tl.run_hmmsearch_2('./hmmer_out/all_protein.txt', Args.hmmer_cutoff,
                                             './hmmer_database/lysin_reported.hmm',
                                             './all_protein_filter.faa')
                tl.run(cmd_5_p)
                find_pfam('./all_protein_filter.faa',  './hmmer_database/lysin_reported.txt')
                
                
                # step 9 Filter sequences without EAD
                cmd_6 = tl.run_hmmsearch('./hmmer_out_EAD/all_protein_filter_hmmer_out_EAD.txt', Args.hmmer_cutoff,
                                         curr_dir_hmmerdb_EAD + hmmer_db_EAD_suffix,
                                         './all_protein_pfam_protein.fasta')
                tl.run(cmd_6)
    
                if not os.path.getsize("./all_protein_pfam_protein.fasta"):
                    raise ValueError('No domain was found and No lysins found!!!')
        
                find_pfam_EAD('./all_protein_pfam_protein.fasta', curr_dir_rpe + reported_lysin_EAD_suffix)
                
        
                # step 9 combine results of CAZY and pfam
                os.system('cat all_protein_pfam_protein_EAD.fasta > pfam_EAD.fasta')
                
                dic_fa_ead = fasta2dict('./pfam_EAD.fasta')
                with open('./pfam_EAD_tmp.fasta','w') as f:
                  for key in dic_fa_ead:
                      if all(f not in dic_fa_ead[key] for f in filters):
                          line = key + '\n' + dic_fa_ead[key] + '\n'
                          f.write(line)
                f.close()
                
                x = float(Args.cdhit_cutoff)
                cdhit_cutoff = str("%.2f" %x)
                if cdhit_cutoff == '1.00':
                    os.system('cat ./pfam_EAD_tmp.fasta > ./pfam_EAD_cdhit.fasta')
                    print('cdhit_cutoff 1.00')
                else:
                    cmd_7 = tl.run_cdhit('./pfam_EAD_tmp.fasta', './pfam_EAD_cdhit.fasta', Args.cdhit_cutoff)
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
                  
                  
                  domain_type = {}
                  f_CBD = open(curr_dir_rpc + reported_lysin_CBD_suffix)
                  for i in f_CBD:
                    item = i.strip()
                    domain_type[item] = 'CBD'
                  
                  f_EAD = open(curr_dir_rpe + reported_lysin_EAD_suffix)
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
                  
                  
                  if Args.ref != '':
                    first_dict = SeqIO.to_dict(SeqIO.parse(open('./putative_lysins.fa'),'fasta'))
                    os.chdir(curr_dir)
                    ref_lysins = os.path.abspath(str(Args.ref))
                    second_dict = SeqIO.to_dict(SeqIO.parse(open(ref_lysins),'fasta'))
                    os.chdir(Args.workdir)
                    
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
                  
                  
                time.sleep(120) 
                os.system('rm -r ./hmmer_database/ ./hmmer_out/ ./hmmer_out_EAD/ ./prokka_result/ ./biolib_results/ ./phage_faa/')
                os.system('rm -r ./pfam_EAD_cdhit*')
                os.remove('./all_protein_filter.faa')
                os.remove('./all_protein.faa')
                os.remove('./all_protein_pfam_protein.fasta')
                os.remove('./all_protein_pfam_protein_EAD.fasta')
                os.remove('./pfam_EAD.fasta')
                os.remove('./pfam_EAD_tmp.fasta')
                os.remove('./all_protein_ut.faa')
                os.remove('./molecular_weight.txt')
                os.remove('./MW_Length.txt') 
                os.remove('./Domain_Info.txt')
                os.system('rm -r ./signaltmp/')
    
        else:
            raise ValueError('Error, please check parameter "--bp"')
            
    elif Args.method == 'rpsblast':
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
            curr_dir = sub.getoutput('pwd')
            os.chdir(Args.workdir)
            
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
            
            
            if Args.prophage_method == 'DBSCAN_SWA':
              if Args.type == '':
                  if Args.dbscan_swa != '':
                    # step 2 DBSCAN-SWA predict prophage
                    if os.path.isdir('./DBSCAN_SWA_out/') == True:
                        pass
                    else:
                        os.mkdir('./DBSCAN_SWA_out/')
                                    
                    for i in os.listdir(curr_dir_target + target_suffix):
                        lis = i.split('.')[:-1]
                        prefix = '.'.join(lis)
                        cmd_2 = tl.run_DBSCAN_SWA(Args.dbscan_swa, curr_dir_target + target_suffix + i,
                                                  './DBSCAN_SWA_out/' + prefix, prefix)
                        tl.run(cmd_2)
                     
                        
                    if os.path.isdir('./orf_ffn/') == True:
                        pass
                    else:
                        os.mkdir('./orf_ffn/')
                        
                    # step 4 phanotate annotates prophage ORFs
                    for i in os.listdir('./DBSCAN_SWA_out/'):
                        for j in os.listdir('./DBSCAN_SWA_out/' + i):
                            j_prefix = '.'.join(j.split('.')[:-1])
                            j_suffix = j.split('.')[-1]
                            if j_suffix == 'faa':
                                os.system('mv %s %s' % ('./DBSCAN_SWA_out/' + i + '/' + j_prefix + '.' + j_suffix,
                                                        './DBSCAN_SWA_out/' + i + '/' + j_prefix + '_tmp.' + j_suffix))
                                f = open('./DBSCAN_SWA_out/' + i + '/' + j_prefix + '_tmp.' + j_suffix)
                                with open('./DBSCAN_SWA_out/' + i + '/' + j_prefix + '.' + j_suffix, 'w') as w:
                                    for m in f:
                                        if m != None:
                                            if m.startswith('>'):
                                                line = '>' + i + ':' + m.strip().split('|')[3].split('_')[0] + '-' + m.strip().split('|')[3].split('_')[1] + ':' + m.strip().split('|')[3].split('_')[2] + ':' + m.strip().split('|')[2] + '\n'
                                                w.write(line)
                                            else:
                                                line = m
                                                w.write(line)
                                w.close()
                                os.remove('./DBSCAN_SWA_out/' + i + '/' + j_prefix + '_tmp.' + j_suffix)
                                    
                                shutil.copyfile('./DBSCAN_SWA_out/' + i + '/' + j, './orf_ffn/' + j)
                
                  else:
                      raise ValueError('please add dbscan-swa.py path!')     
                
              else:
                  raise ValueError('redundant parameters "-t" !')  
              
              
            elif Args.prophage_method == 'phispy':
              if Args.type == 'Bacteria':
                # step 1 prokka annotates ORFs
                if os.path.isdir('./prokka_result/') == True:
                    pass
                else:
                    os.mkdir('./prokka_result/')
                
                name_fna = {}
                type_annotation = Args.type
                for i in os.listdir(curr_dir_target + target_suffix):
                    lis = i.split('.')[:-1]
                    name = '.'.join(lis)
                    suffix = i.split('.')[-1]
                    name_fna[name] = suffix
                    cmd_1 = tl.run_prokka(curr_dir_target + target_suffix + i,
                                      './prokka_result/' + name + '/', name, type_annotation)
                    tl.run(cmd_1)
                    
              else:
                  raise ValueError('please correct annotation type!')
                  
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
      
      
              for i in os.listdir('./phispy_out/'):
                if os.path.isdir('./ppn/' + i) == True:
                  pass
                else:
                  os.mkdir('./ppn/' + i)
      
                fna_suffix = name_fna[i]
                prophage_select('./phispy_out/'+ i + '/' + i + '_prophage_coordinates.tsv',
                                curr_dir_target + target_suffix + i + '.' + fna_suffix,'./ppn/' + i + '/' + i)
      
      
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
              
              
              for name in os.listdir('./orf_ffn/'):
                  f = open('./orf_ffn/' + name)
                  prefix = '.'.join(name.split('.')[:-1])
                  os.system('mv %s %s' % ('./orf_ffn/' + prefix + '.ffn', './orf_ffn/' + prefix + '_tmp.ffn'))
                
                  with open('./orf_ffn/' + prefix + '.ffn', 'w') as w:
                    for jj in f:
                      if jj != None:
                        if jj.startswith('>'):
                          line = '>' + prefix + ':' + jj.strip().split(':')[2].split(' ')[0] + ':' + jj.strip().split(':')[1] + '\n'
                          w.write(line)
                        else:
                          line = jj
                          w.write(line)
                  w.close()
                  os.remove('./orf_ffn/' + prefix + '_tmp.ffn')
                
            
            datasize = []
            for i in os.listdir('./orf_ffn/'):
              stats = os.stat('./orf_ffn/' + i)
              datasize.append(stats.st_size)
              
              
            if all(element==0 for element in datasize) != 0:
              if os.path.isdir('./phispy_out/') == True:
                  os.system('rm -r ./orf_ffn/ ./phispy_out/ ./ppn/ ./prokka_result/')
                  raise ValueError('No prophages ORFs found!')
              else:
                  os.system('rm -r ./orf_ffn/ ./DBSCAN_SWA_out/')
                  raise ValueError('No prophages ORFs found!')
              
       
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
                
                if os.path.isdir('./phispy_out/') == True:
                    os.system('rm -r ./orf_ffn/ ./phispy_out/ ./ppn/ ./prokka_result/')
                else:
                    os.system('rm -r ./orf_ffn/ ./DBSCAN_SWA_out/')
                
                os.remove('./all_protein.faa')
                os.remove('./all_protein_ut.faa')
                os.remove('./all_protein_tmp.txt')
              
              else:
                # step 7 calculate molecular weight
                molecular_weight('./all_protein.faa','./all_protein_filter.faa', float(Args.MWU),float(Args.MWL))
                rewrite_dict = fasta2dict_2('./all_protein_filter.faa')
                
                if os.path.isdir('./rps_input/') == True:
                    pass
                else:
                    os.mkdir('./rps_input/')
                    
                if os.path.isdir('./rps_output/') == True:
                    pass
                else:
                    os.mkdir('./rps_output/')
                    
                if os.path.isdir('./add_rps_output/') == True:
                    pass
                else:
                    os.mkdir('./add_rps_output/')
                      
                for key in rewrite_dict:
                  with open('./rps_input/' + key + '.fasta', 'w') as w:
                      line = '>' + key + '\n' + rewrite_dict[key] + '\n'
                      w.write(line)
                  w.close()
                    
        
                # step 8 rpsblast scan in db
                for i in os.listdir('./rps_input/'):
                  lis = i.split('.')[:-1]
                  name = '.'.join(lis)
                  cmd_5 = tl.run_rpsblast('./rps_input/' + i, float(Args.rpsblast_cutoff), './rps_output/' + name +'.out', Args.rpsblast_db)
                  tl.run(cmd_5)
              
                ead_path = os.path.abspath(Args.EAD_info)
                cbd_path = os.path.abspath(Args.CBD_info)
                
                blast_res = './rps_output/'
                blast_len_res = './add_rps_output/'
                if not os.path.exists(blast_len_res):
                  os.mkdir(blast_len_res)
                
                coverage = Args.rpsblast_coverage
                over_lap = Args.rpsblast_over_lap
                ident = Args.rpsblast_identity
                
                dict_ead, dict_cbd, dict_domain = add_length(ead_path, cbd_path, blast_res, blast_len_res)
                Domain_location_dict, isolates_list, position_info = blast_filter(blast_len_res, coverage)
                
                Domain_location_use_dict, Domain_list_get = Domain_filter(Domain_location_dict,
                                                                          isolates_list,
                                                                          ident, coverage, over_lap)
                
                rpsblast_item = []
                ll_position = []
                with open('./putative_lysin_domain_info.csv','w') as w:
                  for key in Domain_location_use_dict:
                    for i in Domain_location_use_dict[key]:
                      if Domain_location_use_dict[key] != 'No Lysins':
                        if float(i[1]) >= ident and float(i[2]) >= coverage:
                          if i[0] in dict_ead:
                            for k in position_info[i[5]]:
                              if i[0] in k:
                                position = k.split('&')[1]
                                ident_p = k.split('&')[2]
                                ll = i[5] + ',' + i[0] + '(Length:' + dict_ead[i[0]][2] + ')' + ',' + dict_domain[i[0]][0] + '(Length:' + dict_ead[i[0]][2] + ')' + ',' + dict_domain[i[0]][1] + '(Length:' + dict_ead[i[0]][2] + ')' + ',' + 'EAD' + ',' + ident_p + ',' + position +'\n'
                                ll_position.append(ll)
                            rpsblast_item.append(i[5])                       
                          elif i[0] in dict_cbd:
                            for k in position_info[i[5]]:
                              if i[0] in k:
                                position = k.split('&')[1]
                                ident_p = k.split('&')[2]
                                ll = i[5] + ',' + i[0] + '(Length:' + dict_cbd[i[0]][2] + ')' + ',' + dict_domain[i[0]][0] + '(Length:' + dict_cbd[i[0]][2] + ')' + ',' + dict_domain[i[0]][1] + '(Length:' + dict_cbd[i[0]][2] + ')' + ',' + 'CBD' + ',' + ident_p + ',' + position +'\n'
                                ll_position.append(ll)
                                
                  ll_list = list(set(ll_position))           
                  for i in ll_list:
                     line = i
                     w.write(line)            
                w.close()
                
                
                ## remove the sequences without EAD
                with open('./rpsblast_tmp.fasta','w') as w:
                    for key in rewrite_dict:
                        if key in rpsblast_item:
                            line = '>' + key + '\n' + rewrite_dict[key] + '\n'
                            w.write(line)
                w.close()
                
                x = float(Args.cdhit_cutoff)
                cdhit_cutoff = str("%.2f" %x)
                if cdhit_cutoff == '1.00':
                    os.system('cat ./rpsblast_tmp.fasta > ./rpsblast_cdhit.fasta')
                    print('cdhit_cutoff 1.00')
                else:
                    cmd_7 = tl.run_cdhit('./rpsblast_tmp.fasta', './rpsblast_cdhit.fasta', Args.cdhit_cutoff)
                    tl.run(cmd_7)
        
                # step 11 remove TMhelix
                tot = sub.getoutput("grep '>' %s | wc -l" % ('./rpsblast_cdhit.fasta'))
            
                if int(tot) > 100:
                    num_1 = int(tot)//100
                    num_2 = int(tot)%100
                    Split_fa_rps('./rpsblast_cdhit.fasta', tot, num_1, num_2)
                  
                    for i in range(1, int(num_1) + 1):
                       time_sleep = random.uniform(60, 180)
                       time.sleep(time_sleep)
                       cmd_8 = tl.run_deeptmhmm('./rpsblast_cdhit-' + str(i) + '00.fasta')
                       tl.run(cmd_8)
                       
                    os.system('cat ./biolib_results/predicted_topologies.3line* > ./biolib_results/predicted_topologies.line')
                    remove_TMhelix('./biolib_results/predicted_topologies.line','./rpsblast_cdhit.fasta','./putative_lysins.fa')
                  
                else:
                    cmd_8 = tl.run_deeptmhmm('./rpsblast_cdhit.fasta')
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
                  
                  
                  if Args.ref != '':
                    first_dict = SeqIO.to_dict(SeqIO.parse(open('./putative_lysins.fa'),'fasta'))
                    os.chdir(curr_dir)
                    ref_lysins = os.path.abspath(str(Args.ref))
                    second_dict = SeqIO.to_dict(SeqIO.parse(open(ref_lysins),'fasta'))
                    os.chdir(Args.workdir)
                    
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
                    
                            
                  elif Args.ref == '':
                    print('aaaaa')
                    
                    with open('./putative_lysins_info.txt','w') as w:
                      line = 'ID' + '\t' + 'MW' + '\t' + 'Length' + '\t' + 'Domains' + '\t' + 'Signalp' + '\n'
                      w.write(line)
                      for key in dic_info:
                        line = key + '\t' + '\t'.join(dic_info[key][0:2]) + '\t' + dic_info[key][2] + '\t' + dic_info[key][-1] + '\n'
                        w.write(line)
                    w.close()
                      
                else:
                  print(state)
                
                time.sleep(120) 
                if os.path.isdir('./phispy_out/') == True:
                    os.system('rm -r ./rps_input/ ./rps_output/ ./add_rps_output/ ./orf_ffn/ ./phispy_out/ ./ppn/ ./prokka_result/ ./biolib_results/')
                else:
                    os.system('rm -r ./rps_input/ ./rps_output/ ./add_rps_output/ ./orf_ffn/ ./DBSCAN_SWA_out/ ./biolib_results/') 
                os.remove('./all_protein_filter.faa')
                os.remove('./all_protein_tmp.txt')
                os.remove('./all_protein.faa')
                os.remove('./all_protein_ut.faa')
                os.system('rm -r rpsblast_cdhit*')
                os.system('rm -r rpsblast_tmp.fasta')
                os.remove('./putative_lysin_domain_info.csv')
                os.remove('./molecular_weight.txt')
                os.remove('./MW_Length.txt') 
                os.remove('./Domain_Info.txt')
                os.system('rm -r ./signaltmp/')
    
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
    
            if Args.type == 'Viruses':
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
                      
            else:
                raise ValueError('please correct annotation type!')
    
    
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
              os.system('rm -r ./prokka_result/ ./phage_faa/')
              raise ValueError('No phage faa found!')
              
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
                
                os.system('rm -r ./prokka_result/')
                os.remove('./all_protein.faa')
                os.remove('./all_protein_ut.faa')
              
              else:     
                # step 5 calculate molecular weight
                molecular_weight('./all_protein.faa','./all_protein_filter.faa', float(Args.MWU),float(Args.MWL))
                rewrite_dict = fasta2dict_2('./all_protein_filter.faa')
                
                if os.path.isdir('./rps_input/') == True:
                    pass
                else:
                    os.mkdir('./rps_input/')
                    
                if os.path.isdir('./rps_output/') == True:
                    pass
                else:
                    os.mkdir('./rps_output/')
                    
                if os.path.isdir('./add_rps_output/') == True:
                    pass
                else:
                    os.mkdir('./add_rps_output/')
                      
                for key in rewrite_dict:
                  with open('./rps_input/' + key + '.fasta', 'w') as w:
                      line = '>' + key + '\n' + rewrite_dict[key] + '\n'
                      w.write(line)
                  w.close()
                    
        
                # step 8 rpsblast scan in db
                for i in os.listdir('./rps_input/'):
                  lis = i.split('.')[:-1]
                  name = '.'.join(lis)
                  cmd_5 = tl.run_rpsblast('./rps_input/' + i, float(Args.rpsblast_cutoff), './rps_output/' + name +'.out', Args.rpsblast_db)
                  tl.run(cmd_5)
              
                ead_path = os.path.abspath(Args.EAD_info)
                cbd_path = os.path.abspath(Args.CBD_info)
                
                blast_res = './rps_output/'
                blast_len_res = './add_rps_output/'
                if not os.path.exists(blast_len_res):
                  os.mkdir(blast_len_res)
                
                coverage = Args.rpsblast_coverage
                over_lap = Args.rpsblast_over_lap
                ident = Args.rpsblast_identity
                
                dict_ead, dict_cbd, dict_domain = add_length(ead_path, cbd_path, blast_res, blast_len_res)
                Domain_location_dict, isolates_list, position_info = blast_filter(blast_len_res, coverage)
                
                Domain_location_use_dict, Domain_list_get = Domain_filter(Domain_location_dict,
                                                                          isolates_list,
                                                                          ident, coverage, over_lap)
                
                
                rpsblast_item = []
                ll_position = []
                with open('./putative_lysin_domain_info.csv','w') as w:
                  for key in Domain_location_use_dict:
                    for i in Domain_location_use_dict[key]:
                      if Domain_location_use_dict[key] != 'No Lysins':
                        if float(i[1]) >= ident and float(i[2]) >= coverage:
                          if i[0] in dict_ead:
                            for k in position_info[i[5]]:
                              if i[0] in k:
                                position = k.split('&')[1]
                                ident_p = k.split('&')[2]
                                ll = i[5] + ',' + i[0] + '(Length:' + dict_ead[i[0]][2] + ')' + ',' + dict_domain[i[0]][0] + '(Length:' + dict_ead[i[0]][2] + ')' + ',' + dict_domain[i[0]][1] + '(Length:' + dict_ead[i[0]][2] + ')' + ',' + 'EAD' + ',' + ident_p + ',' + position +'\n'
                                ll_position.append(ll)
                            rpsblast_item.append(i[5])                       
                          elif i[0] in dict_cbd:
                            for k in position_info[i[5]]:
                              if i[0] in k:
                                position = k.split('&')[1]
                                ident_p = k.split('&')[2]
                                ll = i[5] + ',' + i[0] + '(Length:' + dict_cbd[i[0]][2] + ')' + ',' + dict_domain[i[0]][0] + '(Length:' + dict_cbd[i[0]][2] + ')' + ',' + dict_domain[i[0]][1] + '(Length:' + dict_cbd[i[0]][2] + ')' + ',' + 'CBD' + ',' + ident_p + ',' + position +'\n'
                                ll_position.append(ll)
                                
                  ll_list = list(set(ll_position))           
                  for i in ll_list:
                     line = i
                     w.write(line)            
                w.close()
                
                
                ## remove the sequences without EAD
                with open('./rpsblast_tmp.fasta','w') as w:
                    for key in rewrite_dict:
                        if key in rpsblast_item:
                            line = '>' + key + '\n' + rewrite_dict[key] + '\n'
                            w.write(line)
                w.close()
                
                
                x = float(Args.cdhit_cutoff)
                cdhit_cutoff = str("%.2f" %x)
                if cdhit_cutoff == '1.00':
                    os.system('cat ./rpsblast_tmp.fasta > ./rpsblast_cdhit.fasta')
                    print('cdhit_cutoff 1.00')
                else:
                    cmd_7 = tl.run_cdhit('./rpsblast_tmp.fasta', './rpsblast_cdhit.fasta', Args.cdhit_cutoff)
                    tl.run(cmd_7)
        
                # step 11 remove TMhelix
                tot = sub.getoutput("grep '>' %s | wc -l" % ('./rpsblast_cdhit.fasta'))
            
                if int(tot) > 100:
                    num_1 = int(tot)//100
                    num_2 = int(tot)%100
                    Split_fa_rps('./rpsblast_cdhit.fasta', tot, num_1, num_2)
                  
                    for i in range(1, int(num_1) + 1):
                       time_sleep = random.uniform(60, 180)
                       time.sleep(time_sleep)
                       cmd_8 = tl.run_deeptmhmm('./rpsblast_cdhit-' + str(i) + '00.fasta')
                       tl.run(cmd_8)
                       
                    os.system('cat ./biolib_results/predicted_topologies.3line* > ./biolib_results/predicted_topologies.line')
                    remove_TMhelix('./biolib_results/predicted_topologies.line','./rpsblast_cdhit.fasta','./putative_lysins.fa')
                  
                else:
                    cmd_8 = tl.run_deeptmhmm('./rpsblast_cdhit.fasta')
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
                  
                  
                  if Args.ref != '':
                    first_dict = SeqIO.to_dict(SeqIO.parse(open('./putative_lysins.fa'),'fasta'))
                    os.chdir(curr_dir)
                    ref_lysins = os.path.abspath(str(Args.ref))
                    second_dict = SeqIO.to_dict(SeqIO.parse(open(ref_lysins),'fasta'))
                    os.chdir(Args.workdir)
                    
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
                    
                            
                  elif Args.ref == '':
                    print('aaaaa')
                    
                    with open('./putative_lysins_info.txt','w') as w:
                      line = 'ID' + '\t' + 'MW' + '\t' + 'Length' + '\t' + 'Domains' + '\t' + 'Signalp' + '\n'
                      w.write(line)
                      for key in dic_info:
                        line = key + '\t' + '\t'.join(dic_info[key][0:2]) + '\t' + dic_info[key][2] + '\t' + dic_info[key][-1] + '\n'
                        w.write(line)
                    w.close()
                      
                else:
                  print(state)
                  
                  
                time.sleep(120) 
                os.system('rm -r ./rps_input/ ./rps_output/ ./add_rps_output/ ./prokka_result/ ./biolib_results/ ./phage_faa/')
                os.system('rm -r rpsblast_cdhit*')
                os.system('rm -r rpsblast_tmp.fasta')
                os.remove('./all_protein.faa')
                os.remove('./all_protein_filter.faa')
                os.remove('./all_protein_ut.faa')
                os.remove('./putative_lysin_domain_info.csv')
                os.remove('./molecular_weight.txt')
                os.remove('./MW_Length.txt') 
                os.remove('./Domain_Info.txt')
                os.system('rm -r ./signaltmp/')
    
        else:
            raise ValueError('Error, please check parameter "--bp"')
