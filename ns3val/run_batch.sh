#!/usr/bin/env bash
# Run the packet-level validation batch: methods x seeds x episodes x load levels.
# usage: run_batch.sh <ns3 build dir> <out dir> [episodes] [jobs]
set -u
NS3=$1; OUT=$2; EPS=${3:-30}; JOBS=${4:-$(nproc)}
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=$(ls $NS3/build/scratch/uavval/ns3.*-uavval-optimized)
export LD_LIBRARY_PATH=$NS3/build/lib
mkdir -p $OUT/pol
for f in $HERE/ns3_export/policies/*.json; do python3 $HERE/json2pol.py $f $OUT/pol/$(basename $f .json).txt; done
for L in 1.0 0.3; do for m in B1 B2 Proposed Hybrid HeurMC; do for s in 1 2 3 4 5 6 7 8; do for e in $(seq 1 $EPS); do
  P=$OUT/pol/${m}_s$s.txt; [ $m = HeurMC ] && P=HeurMC
  o=$OUT/L$L/${m}_s${s}_e$e.csv
  echo "[ -s $o ] || $BIN --policy=$P --trace=$HERE/ns3_export/traces/s${s}_e$e.bin --out=$o --mode=ns3 --seed=$((1000*s+e)) --loadScale=$L"
done; done; done; done > $OUT/jobs.txt
mkdir -p $OUT/L1.0 $OUT/L0.3
echo "$(wc -l < $OUT/jobs.txt) jobs on $JOBS workers"
xargs -P $JOBS -I{} sh -c "{}" < $OUT/jobs.txt
echo "done: $(ls $OUT/L1.0 | wc -l) + $(ls $OUT/L0.3 | wc -l) files"
