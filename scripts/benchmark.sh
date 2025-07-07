#!/bin/bash

# 二进制程序名
BIN=./hgemm_cublas_load

# 案例参数和文件名
cases=(
  "Case1 768 768 768"
  "Case2 128 1024 2048"
  "Case3 128 2048 8192"
  "Case4 512 3072 1024"
  "Case5 512 4096 8192"
  "Case6 3136 576 64"
  "Case7 4096 4096 4096"
  "Case8 1024 16384 16384"
  "Case9 4096 16384 14336"
  "Case10 32768 32768 32768"
)

for case in "${cases[@]}"; do
  read -r name M N K <<< "$case"

  input_file="matrices_${name}_${M}x${N}x${K}.txt"
  output_file="result_${name}_${M}x${N}x${K}.txt"

  echo "Running $name with M=$M N=$N K=$K"
  if [ ! -f "$input_file" ]; then
    echo "Input file $input_file not found, skipping..."
    continue
  fi

  $BIN --input "$input_file" --output "$output_file"

  if [ $? -ne 0 ]; then
    echo "Run failed for $name"
    exit 1
  fi

  echo "Finished $name, output saved to $output_file"
done

echo "All cases processed."
