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

# 输出根目录
output_dir="data/input"
mkdir -p "$output_dir"

for case in "${cases[@]}"; do
  # 拆解变量
  read -r name M N K <<<"$case"

  # 构建子目录路径（如：data/input/Case1_768x768x768）
  case_dir="${output_dir}/${name}_${M}x${N}x${K}"
  mkdir -p "$case_dir"

  echo "📦 Generating binary matrices in $case_dir (M=$M, N=$N, K=$K)"

  # 调用 Python 脚本（已默认生成 .bin 文件）
  python3 tools/matrix_generation.py --M "$M" --N "$N" --K "$K" --outdir "$case_dir"

  if [ $? -ne 0 ]; then
    echo "❌ Failed to generate binary matrices for $name"
    exit 1
  fi
done

echo "✅ All binary matrix folders generated successfully."
