import os

def is_overlap(interval1, interval2):
  return max(interval1[0], interval2[0]) <= min(interval1[1], interval2[1])

f = open('./putative_lysins_info_20.txt')

dict_cbd_new = {}
dict_ead_new = {}
for i in f:
  ii = i.strip().split('\t')[3].split(';')
  name = i.strip().split('\t')[0]
  dict_ead = {}
  dict_cbd = {}
  for j in ii:
    if 'CBD' in j:
      lis_cbd = []
      cbd = j.split('&')[4]
      cbd_posi = j.split('&')[5].split(':')[0]
      
      if name in dict_cbd:
        dict_cbd[name].append(j.split('&')[0] + '&' + cbd + '&' + cbd_posi)
      else:
        lis_cbd.append(j.split('&')[0] + '&' + cbd + '&' + cbd_posi)
        dict_cbd[name] = lis_cbd
      
    if 'EAD' in j:
      lis_ead = []
      ead = j.split('&')[4]
      ead_posi = j.split('&')[5].split(':')[0]
      
      if name in dict_ead:
        dict_ead[name].append(j.split('&')[0] + '&' + ead + '&' + ead_posi)
      else:
        lis_ead.append(j.split('&')[0] + '&' + ead + '&' + ead_posi)
        dict_ead[name] = lis_ead
        
    for key in dict_ead:
      for kk in dict_ead[key]:
        length = int(kk.split('&')[0].split('(')[1].strip(')').split(':')[1])
        start = kk.split('&')[2].split('-')[0]
        end = kk.split('&')[2].split('-')[1]
        inv = int(end) - int(start) + 1
        s = float(kk.split('&')[1])
        id = kk.split('&')[0]
        lis = []
        listt = []
        dict_s = {}
        dict_id = {}
        if inv /length > 0.8:
          ll = (start,end)
          dict_s[str(ll)] = s
          dict_id[str(ll)] = id
          lis.append(ll)
          print(lis)
      
      if len(lis) == 1:
        dict_ead_new[name] = id
      
      else:
        lis_new = []
        for jj in range(0,len(lis)):
          lstt = lis
          del lstt[jj]
          for tt in listt:
            if is_overlap(lis[jj], listt[tt]) == True:
              if dict_s[lis[jj]] > dict_s[listt[tt]]:
                lis_new.append(dict_id[lis[jj]])
              else:
                lis_new.append(dict_id[listt[tt]])
            else:
              for uu in lis:
                lis_new.append(dict_id[uu])
          
        dict_ead_new[name] = lis_new
    
                
    for key in dict_cbd:
      for kk in dict_cbd[key]:
        length = int(kk.split('&')[0].split('(')[1].strip(')').split(':')[1])
        start = kk.split('&')[2].split('-')[0]
        end = kk.split('&')[2].split('-')[1]
        inv = int(end) - int(start) + 1
        s = float(kk.split('&')[1])
        id = kk.split('&')[0]
        lis = []
        listt = []
        dict_s = {}
        dict_id = {}
        if inv /length > 0.8:
          ll = (start,end)
          dict_s[str(ll)] = s
          dict_id[str(ll)] = id
          lis.append(ll)
          print(lis)
      
      if len(lis) == 1:
        dict_cbd_new[name] = id
      
      else:
        lis_new = []
        for jj in range(0,len(lis)):
          lstt = lis
          del lstt[jj]
          for tt in listt:
            if is_overlap(lis[jj], listt[tt]) == True:
              if dict_s[lis[jj]] > dict_s[listt[tt]]:
                lis_new.append(dict_id[lis[jj]])
              else:
                lis_new.append(dict_id[listt[tt]])
            else:
              for uu in lis:
                lis_new.append(dict_id[uu])
          
        dict_cbd_new[name] = lis_new   
        
print(dict_ead_new)
print(dict_cbd_new)
  