import os

a = './putative_lysins_info_20.txt'
b = './ead.txt'
c = './cbd.txt'
coverage = 0.8

def is_overlap(interval1, interval2):
  return max(interval1[0], interval2[0]) <= min(interval1[1], interval2[1])

f = open(a)
next(f)

dict_cbd_new = {}
dict_ead_new = {}
for i in f:
  ii = i.strip().split('\t')[3].split(';')
  name = i.strip().split('\t')[0]
  dict_ead = {}
  dict_cbd = {}
  for j in ii:
    if 'EAD' in j:
      lis_ead = []
      ead = j.split('&')[4]
      ead_posi = j.split('&')[5].split(':')[0]
      
      if name in dict_ead:
        dict_ead[name].append(j.split('&')[2] + '&' + ead + '&' + ead_posi)
      else:
        lis_ead.append(j.split('&')[2] + '&' + ead + '&' + ead_posi)
        dict_ead[name] = lis_ead
    
    if 'CBD' in j:
      lis_cbd = []
      cbd = j.split('&')[4]
      cbd_posi = j.split('&')[5].split(':')[0]
      
      if name in dict_cbd:
        dict_cbd[name].append(j.split('&')[2] + '&' + cbd + '&' + cbd_posi)
      else:
        lis_cbd.append(j.split('&')[2] + '&' + cbd + '&' + cbd_posi)
        dict_cbd[name] = lis_cbd
        
    for key in dict_ead:
      lis = []
      dict_s = {}
      dict_id = {}
      for kk in dict_ead[key]:
        length = int(kk.split('&')[0].split('Length:')[1].strip(')'))
        start = kk.split('&')[2].split('-')[0]
        end = kk.split('&')[2].split('-')[1]
        inv = int(end) - int(start) + 1
        s = float(kk.split('&')[1])
        id = kk.split('&')[0]
        if inv /length > float(coverage):
          ll = (start,end)
          dict_s[str(ll)] = s
          dict_id[str(ll)] = id
          lis.append(ll)
      
      if len(lis) == 1:
        dict_ead_new[name] = [id]
      
      else:
        lis_new = []
        for jj in range(0,len(lis)):
          for n in range(1,len(lis) - jj):
            if is_overlap(lis[jj], lis[jj + n]) == True:
              if dict_s[str(lis[jj])] > dict_s[str(lis[jj + n])]:
                lis_new.append(dict_id[str(lis[jj])])
              else:
                lis_new.append(dict_id[str(lis[jj + n])])
            else:
              for uu in lis:
                lis_new.append(dict_id[str(uu)])
          
        dict_ead_new[name] = list(set(lis_new))
    
                
    for key in dict_cbd:
      lis = []
      dict_s = {}
      dict_id = {}
      for kk in dict_cbd[key]:
        length = int(kk.split('&')[0].split('(')[1].strip(')').split(':')[1])
        start = kk.split('&')[2].split('-')[0]
        end = kk.split('&')[2].split('-')[1]
        inv = int(end) - int(start) + 1
        s = float(kk.split('&')[1])
        id = kk.split('&')[0]
        if inv /length > float(coverage):
          ll = (start,end)
          dict_s[str(ll)] = s
          dict_id[str(ll)] = id
          lis.append(ll)
      
      if len(lis) == 1:
        dict_cbd_new[name] = [id]
      
      else:
        lis_new = []
        for jj in range(0,len(lis)):
          for n in range(1,len(lis) - jj):
            if is_overlap(lis[jj], lis[jj + n]) == True:
              if dict_s[str(lis[jj])] > dict_s[str(lis[jj + n])]:
                lis_new.append(dict_id[str(lis[jj])])
              else:
                lis_new.append(dict_id[str(lis[jj + n])])
            else:
              for uu in lis:
                lis_new.append(dict_id[str(uu)])
          
        dict_cbd_new[name] = list(set(lis_new))  
        

with open(b, 'w') as w:
  for key in dict_ead_new:
    if len(dict_ead_new[key]) == 0:
      line = key + '\t' + 'Coverage not meeting the threshold' + '\n'
      w.write(line)
    else:
      line = key + '\t' + '&'.join(dict_ead_new[key]) + '\n'
      w.write(line) 
w.close()

with open(c, 'w') as w:
  for key in dict_cbd_new:
    if len(dict_cbd_new[key]) == 0:
      line = key + '\t' + 'Coverage not meeting the threshold' + '\n'
      w.write(line)
    else:
      line = key + '\t' + '&'.join(dict_cbd_new[key]) + '\n'
      w.write(line) 
w.close()