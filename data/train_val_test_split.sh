#!/bin/bash

# This script creates a random train, val, test split and of the files/folder under data_root
# The path to data_root is passed as argument to the script and should be any /path/to/featherfiles/

data_root=$1
temp_file=temp.txt
target_dir=temp_target
train_dir=${target_dir}/training/featherfiles/
val_dir=${target_dir}/validation/featherfiles/
test_dir=${target_dir}/test/featherfiles/

n="$(ls -1 ${data_root} | wc -l)"
r=15 # Split ratio 0.15
n=$((n*r/100))
echo "$n"

# Create a randomized list of files/folders
ls -1 ${data_root} | sort --random-sort > ${temp_file}

# Prepare target directory
if test -d ${target_dir};
then
  rm -r ${target_dir}
fi
mkdir ${target_dir}
mkdir ${target_dir}/training/
mkdir ${train_dir}
mkdir ${target_dir}/validation/
mkdir ${val_dir}
mkdir ${target_dir}/test/
mkdir ${test_dir}

# Loop and copy
i=1
while read -r file
do
        if [[ $i -le $n ]];
        then
                cp -r ${data_root}/"${file}" -t ${test_dir}
        elif [[ $i -gt $n && $i -lt $((2*n+1)) ]];
        then
                cp -r ${data_root}/"${file}" -t ${val_dir}
        else
                cp -r ${data_root}/"${file}" -t ${train_dir}
        fi
        i=$((i+1))
done < ${temp_file}

rm ${temp_file}
