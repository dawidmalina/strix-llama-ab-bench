#!/bin/bash
# Run on the benchmark host after scp of ab_all.sh, bench_ab.sh, server_ab.py to /tmp.
chmod 755 /tmp/ab_all.sh /tmp/bench_ab.sh /tmp/server_ab.py
if pgrep -f "llama-bench|llama-server -m|server_ab.py|bench_ab.sh" >/dev/null; then echo "something already running"; exit 0; fi
sudo -n -u lemonade env BENCH_ROUNDS=${BENCH_ROUNDS:-2} ROUNDS=${ROUNDS:-2} setsid nohup /tmp/ab_all.sh > /tmp/ab_all.log 2>&1 < /dev/null &
disown
echo started
