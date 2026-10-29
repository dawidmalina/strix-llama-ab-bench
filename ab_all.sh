#!/bin/bash
# Sequential: llama-bench A/B then server A/B (they share the GPU).
/tmp/bench_ab.sh || { echo "bench stage failed/aborted, skipping server stage"; exit 1; }
python3 /tmp/server_ab.py /tmp/srvreg2_$(date +%Y%m%d_%H%M%S)
