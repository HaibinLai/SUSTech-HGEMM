#!/bin/bash

# 定义测试案例参数：Case编号 M N K
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
  # 读变量
  read -r name M N K <<<"$case"

  # 生成文件名
  filename="data/input/matrices_${name}_${M}x${N}x${K}.txt"

  echo "Generating $filename with M=$M, N=$N, K=$K"

  python3 tools/generation.py --M "$M" --N "$N" --K "$K" --output "$filename"

  if [ $? -ne 0 ]; then
    echo "Failed to generate $filename"
    exit 1
  fi
done

echo "All test matrices generated successfully."
