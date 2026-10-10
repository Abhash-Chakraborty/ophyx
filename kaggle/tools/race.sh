#!/bin/sh
# Two copies of the same training session are queued on different TPU types. When one starts
# running, delete the other so they can never both train the same run (max 24 h).
cd "$(dirname "$0")/../.."; set -a; . ./.env; set +a
A=pycoder-train-seq1; B=pycoder-train-seq1-v6e
st() { nice -n 19 ~/.local/bin/kaggle kernels status abhashchakraborty/$1 2>&1 | grep -o '[A-Z_]*"$' | tr -d '"'; }
note() { curl -s -H "Title: $1" -H "Tags: rocket" -d "$2" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null; }
i=0
while [ $i -lt 720 ]; do
  a=$(st $A); b=$(st $B)
  if [ "$a" = "RUNNING" ] || [ "$a" = "COMPLETE" ]; then
    ~/.local/bin/kaggle kernels delete -y abhashchakraborty/$B; echo "$(date -u +%H:%M) $A started on v5e ($a); deleted $B"; note "training started (TPU v5e)" "deleted the queued v6e copy"; exit 0
  fi
  if [ "$b" = "RUNNING" ] || [ "$b" = "COMPLETE" ]; then
    ~/.local/bin/kaggle kernels delete -y abhashchakraborty/$A; echo "$(date -u +%H:%M) $B started on v6e ($b); deleted $A"; note "training started (TPU v6e)" "deleted the queued v5e copy"; exit 0
  fi
  i=$((i+1)); sleep 120
done
echo "neither started in 24 h: $A=$a $B=$b"
