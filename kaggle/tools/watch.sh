#!/bin/sh
# Poll the given pycoder kernels every 10 min; exit when any status changes (max 14 h).
#   kaggle/tools/watch.sh validate-trainer synth-t4-gen ...
cd "$(dirname "$0")/../.."; set -a; . ./.env; set +a
snap() { for k in "$@"; do printf '%s=%s\n' "$k" "$(nice -n 19 ~/.local/bin/kaggle kernels status abhashchakraborty/pycoder-$k 2>&1 | grep -o '[A-Z_]*"$' | tr -d '"')"; done; }
prev=$(snap "$@"); echo "start $(date -u +%H:%M)"; echo "$prev"
i=0
while [ $i -lt 84 ]; do
  sleep 600; i=$((i+1))
  cur=$(snap "$@")
  if [ "$cur" != "$prev" ]; then echo "changed at $(date -u +%H:%M)"; echo "$cur"; exit 0; fi
done
echo "no change in 14h"; echo "$cur"
