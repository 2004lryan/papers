#!/bin/bash
set -u
PROJ="${ARIS_PROJECT_ROOT:?请先 export ARIS_PROJECT_ROOT 指向项目根}"
SB="$HOME/.cache/rya103_verify"
PY=/opt/miniconda3/bin/python
rm -rf "$SB"
mkdir -p "$SB/02code" "$SB/04outputs/105_seeds" "$SB/05logs"
for f in 54c5_realkinetics 103sensitivity_pilot; do
  cp "$PROJ/02code/$f.py" "$SB/02code/" || { echo "FATAL copy $f"; exit 1; }
done
ln -s "$PROJ/03data" "$SB/03data"
cd "$SB"
for seed in 20060515 20041210 19810915 2023 2024; do
  SEED_OVERRIDE=$seed "$PY" 02code/54c5_realkinetics.py > "$SB/05logs/54_s$seed.out" 2>&1
  echo "54_s$seed rc=$?"
  SEED_OVERRIDE=$seed "$PY" 02code/103sensitivity_pilot.py > "$SB/05logs/103_s$seed.out" 2>&1
  rc=$?
  echo "103_s$seed rc=$rc"
  if [ $rc -ne 0 ]; then tail -5 "$SB/05logs/103_s$seed.out"; fi
  [ -f 04outputs/103sensitivity_pilot.json ] && mv 04outputs/103sensitivity_pilot.json "04outputs/105_seeds/103sensitivity_pilot-seed$seed.json"
  [ -f 04outputs/103sensitivity_pilot-grid.csv ] && mv 04outputs/103sensitivity_pilot-grid.csv "04outputs/105_seeds/103sensitivity_pilot-grid-seed$seed.csv"
done
echo DONE
