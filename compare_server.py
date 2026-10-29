import json, glob, statistics as st, sys
d = sys.argv[1]
R = {}
for f in sorted(glob.glob(f"{d}/*_r?.json")):
    r = json.load(open(f)); R.setdefault(r["tag"], []).append(r)

def med(tag, test, key):
    v = [x[key] for r in R[tag] for x in r["tests"][test] if x.get(key) is not None]
    return st.median(v) if v else float("nan")

print(f"{'test':8} {'metric':6} {'base':>9} {'new':>9} {'diff%':>7}")
for t in ("code", "chat", "long32k"):
    for k, n in (("prompt_per_second", "pp"), ("predicted_per_second", "tg")):
        b, nw = med("base", t, k), med("new", t, k)
        print(f"{t:8} {n:6} {b:9.2f} {nw:9.2f} {100*(nw-b)/b:+7.1f}")
    for tag in ("base", "new"):
        a = sum(x["draft_n_accepted"] or 0 for r in R[tag] for x in r["tests"][t])
        n = sum(x["draft_n"] or 0 for r in R[tag] for x in r["tests"][t])
        print(f"{t:8} accept {tag:>4}: {a}/{n} = {a/n*100 if n else 0:.1f}%")
print()
for t in ("code", "chat", "long32k"):
    hs = {tag: {x["sha256"] for r in R[tag] for x in r["tests"][t]} for tag in R}
    det = {tag: [r["tests"][t + "_deterministic"] for r in R[tag]] for tag in R}
    print(f"{t:8} hashes base={sorted(hs['base'])} new={sorted(hs['new'])} same={hs['base']==hs['new']} deterministic={det}")
for tag in R:
    print(tag, "image_red:", [r["tests"]["image_red"]["ok"] for r in R[tag]], "startup_s:", [r["startup_s"] for r in R[tag]])
