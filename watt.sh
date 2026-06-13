#!/usr/bin/env bash
# Live GPU power for the Intel Arc Pro B60 (xe driver), read straight from the
# hwmon energy counters. No root, no extra packages — pure sysfs.
#
#   ./watt.sh            # sample every 1s
#   ./watt.sh 2          # sample every 2s
#   PCI=0000:03:00.0 ./watt.sh   # override the card (default is the B60)
#
# Ctrl-C prints min/avg/max for the run (handy alongside bench.sh).
set -euo pipefail

PCI=${PCI:-0000:03:00.0}      # Arc Pro B60
interval=${1:-1}              # seconds between samples

# The xe hwmon node has no instantaneous power reading — only cumulative energy
# (microjoules). Locate it by PCI address + driver name, since hwmonN numbering
# is not stable across reboots.
HW=""
for d in /sys/bus/pci/devices/"$PCI"/hwmon/hwmon*; do
  [ -r "$d/name" ] || continue
  [ "$(cat "$d/name")" = "xe" ] && { HW="$d"; break; }
done
[ -n "$HW" ] || { echo "no 'xe' hwmon found under PCI $PCI" >&2; exit 1; }

have_pkg=0; [ -r "$HW/energy2_input" ] && have_pkg=1   # energy2 = pkg (chip)

# running stats (card)
n=0; sum=0; min=999999; max=0
summary() {
  echo
  if [ "$n" -gt 0 ]; then
    printf 'samples=%d  card W: min=%.1f avg=%.1f max=%.1f\n' \
      "$n" "$min" "$(echo "$sum/$n" | bc -l)" "$max"
  fi
  exit 0
}
trap summary INT TERM

c0=$(cat "$HW/energy1_input"); t0=$(date +%s.%N)
[ "$have_pkg" = 1 ] && p0=$(cat "$HW/energy2_input")
printf '%-10s %9s %9s\n' "time" "card(W)" "pkg(W)"

while sleep "$interval"; do
  c1=$(cat "$HW/energy1_input"); t1=$(date +%s.%N)
  dt=$(echo "$t1 - $t0" | bc -l)
  cw=$(echo "scale=2; ($c1-$c0)/1000000/$dt" | bc -l)
  if [ "$have_pkg" = 1 ]; then
    p1=$(cat "$HW/energy2_input")
    pw=$(echo "scale=2; ($p1-$p0)/1000000/$dt" | bc -l)
    p0=$p1
  else
    pw="n/a"
  fi
  printf '%-10s %9.1f %9s\n' "$(date +%H:%M:%S)" "$cw" "$pw"

  # update card stats
  n=$((n+1)); sum=$(echo "$sum + $cw" | bc -l)
  awk "BEGIN{exit !($cw < $min)}" && min=$cw
  awk "BEGIN{exit !($cw > $max)}" && max=$cw
  c0=$c1; t0=$t1
done
