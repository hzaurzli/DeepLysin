import os
import subprocess as sub
import argparse

parser = argparse.ArgumentParser(description="Lysin finder")
parser.add_argument("-i", "--input", required=True, type=str, help="")
parser.add_argument("-t", "--tmp", required=True, type=str, help="")
parser.add_argument("-o", "--output", required=True, type=str, help="")
parser.add_argument("-n", "--num", required=True, type=int, help="")
Args = parser.parse_args()

path = os.path.abspath(Args.input)
path_2 = os.path.abspath(Args.tmp)
path_3 = os.path.abspath(Args.output)

tot = sub.getoutput("ls -r %s | wc -l" % (path + '/*'))
print(tot)

if os.path.isdir('./data/') == True:
    pass
else:
    os.mkdir('./data/')
    
if os.path.isdir('./datatmp/') == True:
    pass
else:
    os.mkdir('./datatmp/')

if os.path.isdir('./result/') == True:
    pass
else:
    os.mkdir('./result/')

lis = []
for i in os.listdir(path):
  lis.append(i)


nn = int(Args.num)
if int(tot) > nn:
  num_1 = int(tot)//nn
  print(num_1)
  num_2 = int(tot)%nn
  print(num_2)
  
  for n in range(1,int(num_1) + 1):
    lis2 = lis[(n-1)*nn:(n-1)*nn + nn]
    if os.path.isdir(path_2 + '/data_' + str(n) + '/') == True:
          pass
    else:
        os.mkdir(path_2 + '/data_' + str(n) + '/')
    
    if os.path.isdir(path_3 + '/res_' + str(n) + '/') == True:
       pass
    else:   
       os.mkdir(path_3 + '/res_' + str(n) + '/')
    for m in lis2:
      os.system('cp %s %s' % (path + '/' + m, path_2 + '/data_' + str(n) + '/'))
      
  if num_2 != 0:
     lis2 = lis[num_1*nn:num_1*nn+num_2]
     
     if os.path.isdir(path_2 + '/data_' + str(num_1 + 1) + '/') == True:
         pass
     else:
         os.mkdir(path_2 + '/data_' + str(num_1 + 1) + '/')
         
     if os.path.isdir(path_3 + '/res_' + str(num_1 + 1) + '/') == True:
         pass
     else:   
         os.mkdir(path_3 + '/res_' + str(num_1 + 1) + '/')
     
     for k in lis2:
         os.system('cp %s %s' % (path + '/' + k, path_2 + '/data_' + str(num_1 + 1) + '/'))
  else:
      pass
      
else:
    os.system('cp %s %s' % (path + '/' + '*', path_2 + '/data_0/'))
  