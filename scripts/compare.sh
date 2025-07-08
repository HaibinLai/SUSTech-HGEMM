#!/bin/bash

# 二进制程序名（包含 cuBLAS 和 kernel 比较功能）
BIN=build/hgemm_compare

# 输入数据根目录，输出结果根目录
INPUT_ROOT="data/input"
OUTPUT_ROOT="data/output/compare_results"

# 测试案例参数（可自行扩展）
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

# 创建输出目录
mkdir -p "$OUTPUT_ROOT"

for case in "${cases[@]}"; do
  read -r name M N K <<< "$case"

  input_dir="${INPUT_ROOT}/${name}_${M}x${N}x${K}"
  output_file="${OUTPUT_ROOT}/compare_result_${name}_${M}x${N}x${K}.txt"

  if [ ! -d "$input_dir" ]; then
    echo "❌ Input directory $input_dir not found, skipping $name ..."
    continue
  fi

  echo "🚀 Running $name with M=$M N=$N K=$K"

  # 执行比较程序
  $BIN --indir "$input_dir" --outdir "$OUTPUT_ROOT"
  if [ $? -ne 0 ]; then
    echo "❌ Run failed for $name"
    exit 1
  fi

  echo "✅ Finished $name, output saved to $output_file"
  echo ""
done

echo "🎉 All GEMM comparison cases processed."
