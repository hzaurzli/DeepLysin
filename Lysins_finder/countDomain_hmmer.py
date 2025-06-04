import os

a = './putative_lysins_info.txt'
b = './domain.txt'
coverage = 0.8

def is_overlap(interval1, interval2):
  return max(interval1[0], interval2[0]) <= min(interval1[1], interval2[1])

f = open(a)
next(f)

dict_domain_new = {}
for i in f:
  ii = i.strip().split('\t')[3].split(';')
  name = i.strip().split('\t')[0]
  dict_domain = {}
  for j in ii:   
    lis_domain = []
    domain = j.split('&')[2]
    domain_posi = j.split('&')[3]
    
    if name in dict_domain:
      dict_domain[name].append(j.split('&')[0] + '&' + domain + '&' + domain_posi)
    else:
      lis_domain.append(j.split('&')[0] + '&' + domain + '&' + domain_posi)
      dict_domain[name] = lis_domain
        
    for key in dict_domain:
      lis = []
      dict_s = {}
      dict_id = {}
      for kk in dict_domain[key]:
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
        dict_domain_new[name] = [id]
      
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
        
        dict_domain_new[name] = list(set(lis_new))
        

with open(b, 'w') as w:
  for key in dict_domain_new:
    if len(dict_domain_new[key]) == 0:
      line = key + '\t' + 'Coverage not meeting the threshold' + '\n'
      w.write(line)
    else:
      line = key + '\t' + '&'.join(dict_domain_new[key]) + '\n'
      w.write(line)
w.close()
