---
name: llama-build-ab
description: 'Regression A/B test of a new llama.cpp fork build vs baseline on the Strix Halo host. Use when the user wants to check a new build for performance or correctness regressions (t/s, MTP acceptance, output hashes), rebuild beta-strix-llama.cpp, or rerun/compare the A/B benchmarks.'
argument-hint: 'optional: skip build / only bench / only server stage'
---

# llama.cpp build A/B regression

## Fixed setup
- Host: `ssh $AB_HOST` (`AB_HOST=user@host`); run tests as `sudo -n -u lemonade` (passwordless sudo). The SSH user cannot read `/var/lib/lemonade/lab`.
- Baseline: `/var/lib/lemonade/lab/rocm-strix-llama.cpp` (do not rebuild unless asked).
- Candidate: `/var/lib/lemonade/lab/beta-strix-llama.cpp` (latest code, pulled by the user).
- Model/flags: Qwen3.8-Flash-Next UD-Q4_K_XL + MTP draft, `-b 4096 -ub 4096 -fa on -lzm on-direct -ngl 999`.
- Scripts: this repo root (bench_ab.sh, server_ab.py, ab_all.sh, launch.sh, compare_bench.py, compare_server.py, build_candidate.sh).

## Procedure
1. Ask the user to confirm before building or launching (GPU/host impact). Check the GPU is free: `pgrep -af "llama-|server_ab|bench_ab"`.
2. Verify the candidate source: `git -c safe.directory=$D -C $D log -1 --oneline`.
3. Build only if needed: `build_candidate.sh` (fresh `cmake -B build`, `-DCMAKE_BUILD_RPATH='$ORIGIN'`). Never copy a `build/` dir from another checkout: its CMakeCache and RUNPATH point to the original and the old code gets built/loaded.
4. Verify the build: `LD_LIBRARY_PATH=$D/build/bin $D/build/bin/llama-server --version` must show the candidate commit, and `ldd` must resolve `libllama.so` inside `$D/build/bin`. The scripts abort on mismatch.
5. Copy and launch detached (the user may close the laptop):
   ```bash
   scp ab_all.sh bench_ab.sh server_ab.py "$AB_HOST":/tmp/
   ssh -o BatchMode=yes "$AB_HOST" 'bash -s' < launch.sh
   ```
   Stage 1 (llama-bench A/B/A/B) runs, then stage 2 (llama-server) on port 18001. They share the GPU, so never run them in parallel.
6. Check progress with `cat /tmp/regression_*/run.log` and `/tmp/srvreg2_*/run.log`; `DONE` marks the end of each stage. Do not poll in a loop.
7. Fetch results and compare:
   ```bash
   scp -r "$AB_HOST":/tmp/regression_<date> /tmp/ab
   scp -r "$AB_HOST":/tmp/srvreg2_<date> /tmp/srv
   python3 compare_bench.py /tmp/ab && python3 compare_server.py /tmp/srv
   ```

## Interpreting results
- Regression: `tg` drop > 3%, draft acceptance lower by > 5 pp, output hashes differing from baseline, image test failing, any crash/OOM/ROCm error.
- Short-prompt `pp` (pp512, pp4096) varies 5-15% between runs: treat as noise, do not flag.
- `chat`/`long32k` are non-deterministic at temp 0 in both builds; compare hash sets between builds, not within one build.
- Not covered: ~128k context fill and long-session memory leaks.

## Pitfalls
- Terminal heredocs (`cat <<EOF`) get garbled in the VS Code terminal; write files with the file tools, then `scp`/`ssh 'bash -s' < file`.
- Do not use `rm -rf` on build dirs without user approval; move aside (`mv build build.old`).
- Do not write tokens or secrets into results or memory.
