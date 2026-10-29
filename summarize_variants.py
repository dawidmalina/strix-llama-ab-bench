#!/usr/bin/env python3
"""Summarize beta knob sweep (MTP draft length / lazy mode) next to the n=2 on-direct reference."""
import glob, json, statistics as st, sys

ref = sys.argv[1]
var = sys.argv[2]
runs = {"n2 on-direct r1": f"{ref}/new_r1.json", "n2 on-direct r2": f"{ref}/new_r2.json"}
for v in ("n3_on-direct", "n4_on-direct", "n2_off"):
    runs[v] = glob.glob(f"{var}/*_{v}/new_r1.json")[0]

print(f"{'variant':17}{'test':9}{'tg':>8}{'pp':>9}{'accept':>9}  hashes")
for k, p in runs.items():
    r = json.load(open(p))
    if not r.get("ok"):
        print(k, "FAILED", r.get("error"))
        continue
    for t in ("code", "chat", "long32k"):
        x = r["tests"][t]
        tg = st.median(i["predicted_per_second"] for i in x)
        pp = st.median(i["prompt_per_second"] for i in x)
        d = sum(i["draft_n"] for i in x)
        a = sum(i["draft_n_accepted"] for i in x)
        print(f"{k:17}{t:9}{tg:8.2f}{pp:9.1f}{100 * a / d:8.1f}%  {sorted({i['sha256'][:6] for i in x})}")
    print(f"{k:17}image={r['tests']['image_red']['ok']} startup={r['startup_s']}s")
