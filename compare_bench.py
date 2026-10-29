import json, glob, statistics, sys
from collections import defaultdict

d = sys.argv[1]
res = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{d}/*_r*.json")):
    tag = f.split("/")[-1].split("_")[0]
    for r in json.load(open(f)):
        key = (r["n_prompt"], r["n_gen"], r.get("n_depth", 0))
        res[key][tag].append(r["avg_ts"])

print(f"{'test':22} {'base':>9} {'new':>9} {'diff%':>7}")
for (p, g, dep), v in sorted(res.items()):
    name = (f"pp{p}" if p else f"tg{g}") + (f" @d{dep}" if dep else "")
    b = statistics.median(v["base"]) if v["base"] else float("nan")
    n = statistics.median(v["new"]) if v["new"] else float("nan")
    print(f"{name:22} {b:9.2f} {n:9.2f} {100*(n-b)/b:+7.1f}   runs b/n={v['base']} / {v['new']}")
