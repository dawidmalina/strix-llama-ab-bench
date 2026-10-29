#!/bin/bash
# Usage: cmp_remote.sh <host result dir>  (compares its new_r1.json with /tmp/srv/base_r1.json)
n=$(basename "$1")
rm -rf /tmp/cmpx_$n /tmp/cmpd_$n
scp -rq "${AB_HOST:?set AB_HOST=user@host}":"$1" /tmp/cmpx_$n && mkdir -p /tmp/cmpd_$n && cp /tmp/srv/base_r1.json /tmp/cmpd_$n/ && cp /tmp/cmpx_$n/new_r1.json /tmp/cmpd_$n/
cd "$(dirname "$0")" && python3 compare_server.py /tmp/cmpd_$n 2>&1 | grep -E "^(code|chat|long32k) +(tg|accept  new)"
