#!/bin/bash

# binary program name
BIN=build/hgemm_cublas

# Input data root directory, output result root directory
INPUT_ROOT="data/input"
OUTPUT_ROOT="data/output/cublas_results"

# Test case parameters
cases=(
  "Case1 768 768 768"
  "Case2 128 1024 2048"
  "Case3 128 2048 8192"
)

mkdir -p "$OUTPUT_ROOT"

for case in "${cases[@]}"; do
  read -r name M N K <<< "$case"

  input_dir="${INPUT_ROOT}/${name}_${M}x${N}x${K}"
  output_file="${OUTPUT_ROOT}/cublas_result_${name}_${M}x${N}x${K}.txt"

  if [ ! -d "$input_dir" ]; then
    echo "❌ Input directory $input_dir not found, skipping $name ..."
    continue
  fi

  echo "🚀 Running $name with M=$M N=$N K=$K"

  $BIN --indir "$input_dir" --outdir "$OUTPUT_ROOT"

  if [ $? -ne 0 ]; then
    echo "❌ Run failed for $name"
    exit 1
  fi

  echo "✅ Finished $name, output saved to cublas_result_${name}_${M}x${N}x${K}.txt"
  echo ""
done

echo "🎉 All benchmark cases processed."
