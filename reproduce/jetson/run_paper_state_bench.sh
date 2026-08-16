#!/bin/bash
# Paper-era machine-state emulation driver (no root required).
#
# The paper's Fig 2/3 sessions ran with busy CPU cores in the 1344 MHz band,
# raised there by desktop background activity through schedutil (known from
# surviving 2026-05-28 tegrastats telemetry).
# This driver rebuilds that state with calibrated loads and runs the shipped
# benchmark, sampling both cpufreq policies throughout as evidence of the
# achieved state.
#
# Usage:  ./run_paper_state_bench.sh <latency|throughput> <A|Aprime|App> <round-id>
#
#   A      : one background worker per cpufreq policy (duty 0.60, cpu3 + cpu5)
#            -> both policies sit in the 1344 MHz band.
#   Aprime : A + fragmented preemption load (10 procs, ~2000 wakeups/s, unpinned)
#            -> emulates the desktop session's scheduling footprint.
#   App    : A + stronger preemption load pinned to the benchmark cores
#            (14 procs on cpu0-2,4) -> closest simultaneous match of the
#            Fig 2(a) column (floor within 2%, XGBoost overhead 32-34 ms).
#
# After every run the result JSON is copied into state_emulation_results/
# (jetson/benchmark/results/ keeps this run's output; both are gitignored).
set -u
MODE=${1:?latency|throughput}
CFG=${2:?A|Aprime|App}
ROUND=${3:-1}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
RES=$REPO/jetson/benchmark/results
OUT=$HERE/state_emulation_results
mkdir -p "$OUT"

pkill -f 'bg_load[.]py'; pkill -f 'churn_load[.]py'; sleep 6
setsid taskset -c 3 python3 "$HERE/bg_load.py" --cores 1 --duty 0.60 --mb 0 >/dev/null 2>&1 &
setsid taskset -c 5 python3 "$HERE/bg_load.py" --cores 1 --duty 0.60 --mb 0 >/dev/null 2>&1 &
case "$CFG" in
  Aprime) setsid python3 "$HERE/churn_load.py" >/dev/null 2>&1 & ;;
  App)    setsid taskset -c 0-2,4 python3 "$HERE/churn_load.py" --procs 14 >/dev/null 2>&1 & ;;
esac
sleep 8

TAG=${MODE}_${CFG}_r${ROUND}
( while :; do
    paste -d' ' /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq \
                /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq
    sleep 0.2
  done ) > "$OUT/freq_$TAG.log" 2>/dev/null &
SAMPLER=$!

SCRIPT=run_latency.py; PERF=latency.json
[ "$MODE" = throughput ] && { SCRIPT=run_throughput.py; PERF=throughput.json; }
ok=0
for attempt in 1 2 3; do
  echo "=== $TAG attempt $attempt ==="
  docker exec -w /workspace byd-jetson python3 "jetson/benchmark/$SCRIPT" \
      > "$OUT/stdout_$TAG.log" 2>&1
  grep -q 'Summary' "$OUT/stdout_$TAG.log" && { ok=1; break; }
  echo "  (known startup abort, retrying)"
done
kill $SAMPLER 2>/dev/null
[ $ok -eq 1 ] && cp "$RES/$PERF" "$OUT/${MODE}_${CFG}_r${ROUND}.json"
echo "note: this run overwrote $RES/$PERF (a copy is in $OUT)"
rm -f "$RES/timing.json" "$RES/timing_throughput.json" "$RES/resources_inproc.json"
pkill -f 'bg_load[.]py'; pkill -f 'churn_load[.]py'
echo "$TAG ok=$ok ; achieved frequency (policy0 policy4, top 3):"
sort "$OUT/freq_$TAG.log" | uniq -c | sort -rn | head -3
grep -A10 'Summary' "$OUT/stdout_$TAG.log" | head -12
