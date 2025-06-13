#!/bin/bash
source ~/mambaforge/bin/activate deeplysin;
python ./split_data.py -i ./data -t ./datatmp -o ./result -n 50;

var1=1;
var2=$(ls ./datatmp/ | grep "^d" | wc -l)

for i in `seq $var1 $var2`;
do 
  python ./lysins_finder_super.py -p ./datatmp/data_$i -wkdir ./result/res_$i -bp B -pp DBSCAN_SWA -ds /home/user/YCH/deeplysin/DBSCAN-SWA/bin/dbscan-swa.py -c 1.00 -ml 10 -mu 5000000000000000 -EI /home/user/CR/database/cdd/EAD_info.csv -CI /home/user/CR/database/cdd/CBD_info.csv -rpsdb /home/user/CR/database/cdd/pldb/pldb -m rpsblast -rident 20 -rc 0.01;
done;
