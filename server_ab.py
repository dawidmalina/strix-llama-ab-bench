#!/usr/bin/env python3
"""Server-layer A/B regression: speed, MTP acceptance, determinism, long ctx, image smoke."""
import base64, hashlib, json, os, signal, struct, subprocess, sys, time, urllib.request, zlib

OUT = sys.argv[1]
PORT = 18001
HUB = "/home/lemonade/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots"
S1 = f"{HUB}/766911a6b7369840a91dbcd95f9f997acaab6cd6"
MODEL = f"{S1}/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
MMPROJ = f"{S1}/mmproj-BF16.gguf"
DRAFT = f"{HUB}/38bb39ee97821de2c9009abb7e93950eec396e66/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
BUILDS = {"base": "/var/lib/lemonade/lab/rocm-strix-llama.cpp",
          "new": os.environ.get("NEW_ROOT", "/var/lib/lemonade/lab/beta-strix-llama.cpp")}
ROUNDS = int(os.environ.get("ROUNDS", "2"))
REPS = 3
SPEC_N = os.environ.get("SPEC_N", "2")
LZM = os.environ.get("LZM", "on-direct")
TAGS = os.environ.get("TAGS", "base,new").split(",")
# WRAP: command prefix for the server, {tag}/{rnd} expanded (e.g. rocprofv3 ... -d /tmp/prof_{tag} --); ONLY: comma list of prompts, skips image
WRAP = os.environ.get("WRAP", "")
ONLY = [p for p in os.environ.get("ONLY", "").split(",") if p]


def server_cmd(root, tag="", rnd=0):
    return WRAP.format(tag=tag, rnd=rnd).split() + [
        f"{root}/build/bin/llama-server", "-m", MODEL, "--ctx-size", "131072", "--port", str(PORT),
        "--jinja", "--metrics", "--mmproj", MMPROJ, "--model-draft", DRAFT,
        "--chat-template-file", "/home/lemonade/chat_template.jinja",
        "--chat-template-kwargs", '{"reasoning_effort": "medium"}',
        "--min-p", "0.0", "--n-predict", "8192", "--presence-penalty", "0.0",
        "--reasoning-format", "deepseek", "--repeat-penalty", "1.0",
        "--spec-draft-n-max", SPEC_N, "--spec-type", "draft-mtp,ngram-mod",
        "--temp", "1.0", "--top-k", "20", "--top-p", "0.95",
        "-b", "4096", "-fa", "on", "-lzm", LZM, "-ngl", "999", "-ub", "4096", "--parallel", "1",
    ]


def http(path, body=None, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def png_solid(w, h, rgb):
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


FILLER = ("The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
          "Sphinx of black quartz, judge my vow. ")
PROMPTS = {
    "code": "Write a Python function that merges overlapping intervals and explain its complexity.",
    "chat": "Explain in three short paragraphs why the sky is blue.",
    "long32k": FILLER * int(os.environ.get("LONG_N", "1600")) + "\nQuestion: summarize the repeated text above in one sentence.\nAnswer:",
}


def completion(prompt, n=256):
    t = time.time()
    r = json.loads(http("/completion", {"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1,
                                        "seed": 1, "cache_prompt": False}))
    tm = r.get("timings", {})
    return {
        "wall_s": round(time.time() - t, 2),
        "prompt_n": tm.get("prompt_n"), "prompt_per_second": tm.get("prompt_per_second"),
        "predicted_n": tm.get("predicted_n"), "predicted_per_second": tm.get("predicted_per_second"),
        "draft_n": tm.get("draft_n"), "draft_n_accepted": tm.get("draft_n_accepted"),
        "sha256": hashlib.sha256(r["content"].encode()).hexdigest()[:16],
        "head": r["content"][:120],
    }


def verify(root):
    exe = f"{root}/build/bin/llama-server"
    env = dict(os.environ, LD_LIBRARY_PATH=f"{root}/build/bin")
    head = subprocess.check_output(["git", "-c", f"safe.directory={root}", "-C", root, "rev-parse", "--short=9", "HEAD"], text=True).strip()
    ver = subprocess.run([exe, "--version"], env=env, capture_output=True, text=True)
    out = ver.stdout + ver.stderr
    ldd = subprocess.run(["ldd", exe], env=env, capture_output=True, text=True).stdout
    lib = next((l.split()[2] for l in ldd.splitlines() if "libllama.so" in l), "")
    ok = head in out and lib.startswith(root + "/")
    print(f"verify {root}: head={head} lib={lib} ok={ok}", flush=True)
    if not ok:
        sys.exit(f"ABORT: {root} version/lib mismatch")


def run_build(tag, rnd):
    root = BUILDS[tag]
    log = open(f"{OUT}/{tag}_r{rnd}.server.log", "w")
    env = dict(os.environ, LD_LIBRARY_PATH=f"{root}/build/bin")
    proc = subprocess.Popen(server_cmd(root, tag, rnd), stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    res = {"tag": tag, "round": rnd, "ok": False}
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                res["error"] = f"server exited rc={proc.returncode}"
                return res
            try:
                if json.loads(http("/health", timeout=5)).get("status") == "ok":
                    break
            except Exception:
                pass
            if time.time() - t0 > 1800:
                res["error"] = "startup timeout"
                return res
            time.sleep(5)
        res["startup_s"] = round(time.time() - t0, 1)
        completion("Hello", 16)  # warmup
        res["tests"] = {}
        for name, p in PROMPTS.items():
            if ONLY and name not in ONLY:
                continue
            res["tests"][name] = [completion(p) for _ in range(REPS)]
            hashes = {x["sha256"] for x in res["tests"][name]}
            res["tests"][name + "_deterministic"] = len(hashes) == 1
        if ONLY:
            res["ok"] = True
            return res
        img = base64.b64encode(png_solid(64, 64, (255, 0, 0))).decode()
        r = json.loads(http("/v1/chat/completions", {
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}},
                {"type": "text", "text": "What is the dominant color of this image? Answer with one word."}]}],
            "max_tokens": 512, "temperature": 0}))
        txt = (r["choices"][0]["message"].get("content") or "") + (r["choices"][0]["message"].get("reasoning_content") or "")
        res["tests"]["image_red"] = {"ok": "red" in txt.lower(), "text": txt[-200:]}
        res["metrics_tail"] = [l for l in http("/metrics").splitlines() if l and not l.startswith("#")][:40]
        res["ok"] = True
    except Exception as e:
        res["error"] = repr(e)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # server already exited (e.g. rejected a CLI flag)
        log.close()
        time.sleep(10)
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    for root in BUILDS.values():
        verify(root)
    for rnd in range(1, ROUNDS + 1):
        for tag in TAGS:
            print(f"=== {tag} r{rnd} {time.strftime('%T')}", flush=True)
            r = run_build(tag, rnd)
            json.dump(r, open(f"{OUT}/{tag}_r{rnd}.json", "w"), indent=1)
            print(f"ok={r['ok']} {r.get('error', '')}", flush=True)
    open(f"{OUT}/DONE", "w").close()
    print("DONE", time.strftime("%T"), flush=True)


main()
