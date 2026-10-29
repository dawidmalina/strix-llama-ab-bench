#!/bin/bash
# Waits for the variants sweep to finish, then runs the full A/B with more rounds. Run as lemonade, detached.
while pgrep -f "variants.sh|server_ab.py|llama-server -m|llama-bench" >/dev/null; do sleep 20; done
BENCH_ROUNDS=5 ROUNDS=4 /tmp/ab_all.sh
