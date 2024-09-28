#!/bin/bash

data_dir="../../../Data"
output_dir="../data/"

default_pub="${data_dir}/public/pub_domains_20201029"
default_alx="${data_dir}/alexa"

echo "Starting"

for file in "$@"
do
echo $file
left_trimed=${file##*/}
trimed=${left_trimed%%_*}

vt_file="${data_dir}/vt/${trimed}_mal_firstseen.csv"
vt_mal_file="${output_dir}/${trimed}_mal_firstseen_malicious.csv"
vt_mal_no_pub_file="${output_dir}/${trimed}_mal_firstseen_malicious_no_public.csv"

python3 filter_1_malicious_vt_urls.py -i $vt_file -o $output_dir

python3 filter_2_public_domains.py -i $vt_mal_file -p $default_pub -o $output_dir

python3 filter_3_alexa_domains.py -i $vt_mal_no_pub_file -a $default_alx -o $output_dir
done

# rm $vt_mal_file
# rm $vt_mal_no_pub_file

echo "Done!"