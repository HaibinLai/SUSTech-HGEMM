#!/bin/bash

# program
BIN=build/hgemm_custom

INPUT_ROOT="data/input"
OUTPUT_ROOT="data/output/custom_results"
NCU_REPORT_ROOT="data/ncu_reports"

# Test cases
cases=(
  "Case7 4096 4096 4096"
)

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$NCU_REPORT_ROOT"

for case in "${cases[@]}"; do
  read -r name M N K <<< "$case"

  input_dir="${INPUT_ROOT}/${name}_${M}x${N}x${K}"
  output_file="${OUTPUT_ROOT}/custom_result_${name}_${M}x${N}x${K}.txt"
  ncu_report="${NCU_REPORT_ROOT}/ncu_report_${name}_${M}x${N}x${K}.qdrep"

  if [ ! -d "$input_dir" ]; then
    echo "❌ Input directory $input_dir not found, skipping $name ..."
    continue
  fi

  echo "🚀 Profiling $name with M=$M N=$N K=$K using ncu"

  ncu --set full --target-processes all --output "$ncu_report" \
      $BIN --indir "$input_dir" --outdir "$OUTPUT_ROOT"

  if [ $? -ne 0 ]; then
    echo "❌ Profiling failed for $name"
    exit 1
  fi

  echo "✅ Finished profiling $name, report saved to $ncu_report"
  echo ""
done

echo "🎉 All benchmark cases profiled with ncu."
