#!/usr/bin/env bash
# Extension batch: cells x load levels x methods x seeds x episodes.
# usage: run_batch2.sh <ns3 dir> <out dir> [jobs]
#   A10 at load 0.05 (30 episodes); A5, A20, B10, C10 at loads 1.0, 0.3, 0.05 (10 episodes)
set -u
NS3=$1; OUT=$2; JOBS=${3:-$(nproc)}
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=$(ls $NS3/build/scratch/uavval/ns3.*-uavval-optimized)
export LD_LIBRARY_PATH=$NS3/build/lib
mkdir -p $OUT/pol
for f in $HERE/ns3_export/policies/*.json; do python3 $HERE/json2pol.py $f $OUT/pol/A10_$(basename $f .json).txt; done
for c in A5 A20 B10 C10; do for f in $HERE/ns3_export/policies/$c/*.json; do python3 $HERE/json2pol.py $f $OUT/pol/${c}_$(basename $f .json).txt; done; done
: > $OUT/jobs2.txt
emit() { # cell scen N load eps tracedir
  local c=$1 sc=$2 n=$3 L=$4 E=$5 TD=$6
  mkdir -p $OUT/$c/L$L
  for m in B1 B2 Proposed Hybrid HeurMC; do for s in 1 2 3 4 5 6 7 8; do for e in $(seq 1 $E); do
    P=$OUT/pol/${c}_${m}_s$s.txt; [ $m = HeurMC ] && P=HeurMC
    o=$OUT/$c/L$L/${m}_s${s}_e$e.csv
    echo "[ -s $o ] || $BIN --policy=$P --trace=$TD/s${s}_e$e.bin --out=$o --mode=ns3 --seed=$((1000*s+e)) --loadScale=$L --scenario=$sc --N=$n" >> $OUT/jobs2.txt
  done; done; done
}
emit A10 A 10 0.05 30 $HERE/ns3_export/traces
for L in 1.0 0.3 0.05; do
  emit A5 A 5 $L 10 $HERE/ns3_export/traces/A5
  emit A20 A 20 $L 10 $HERE/ns3_export/traces/A20
  emit B10 B 10 $L 10 $HERE/ns3_export/traces/B10
  emit C10 C 10 $L 10 $HERE/ns3_export/traces/C10
done
# longest jobs first (B10 and A20 at nominal load)
grep -E "B10/L1.0|A20/L1.0" $OUT/jobs2.txt > $OUT/jobs2s.txt; grep -v -E "B10/L1.0|A20/L1.0" $OUT/jobs2.txt >> $OUT/jobs2s.txt
echo "$(wc -l < $OUT/jobs2s.txt) jobs on $JOBS workers"
xargs -P $JOBS -I{} sh -c "{}" < $OUT/jobs2s.txt
echo "done: $(find $OUT -name '*.csv' | wc -l) files"
