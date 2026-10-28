#!/bin/sh
# Poll the given Kaggle kernels every 10 min; exit when any status changes (max 14 h).
#   kaggle/tools/watch.sh ophyx-bench-gen-big7 ophyx-grpo-p4 ...
[ $# -gt 0 ] || { echo "usage: $0 <kernel-slug>..." >&2; exit 2; }
cd "$(dirname "$0")/../.."; if [ -f .env ]; then set -a; . ./.env; set +a; fi
U=${KAGGLE_USERNAME:-abhashchakraborty}
snap() { for k in "$@"; do printf '%s=%s\n' "$k" "$(nice -n 19 kaggle kernels status "$U/$k" 2>&1 | grep -o '[A-Z_]*"$' | tr -d '"')"; done; }
prev=$(snap "$@"); echo "start $(date -u +%H:%M)"; echo "$prev"
i=0
while [ $i -lt 84 ]; do
  sleep 600; i=$((i+1))
  cur=$(snap "$@")
  if [ "$cur" != "$prev" ]; then echo "changed at $(date -u +%H:%M)"; echo "$cur"; exit 0; fi
done
echo "no change in 14h"; echo "$cur"; exit 1
