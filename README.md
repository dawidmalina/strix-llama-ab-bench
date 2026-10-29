# strix-llama-ab-bench

Scripts for A/B regression testing of a [strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp) build against a baseline on a Strix Halo (gfx1151) host. The metrics are decode speed, MTP draft acceptance and output hashes.

## Setup

- Baseline: `rocm-strix-llama.cpp` at `6aa81b88e`.
- Candidate: any other build, selected with `NEW_ROOT` (default `beta-strix-llama.cpp`).
- Model: Qwen3.8-Flash-Next UD-Q4_K_XL with a Q8_0 MTP draft.
- Paths (`/var/lib/lemonade/lab/...`, `/home/lemonade/...`) and the model snapshot hash are hard-coded in the scripts. Adjust them for your host.

## Main scripts

- `server_ab.py <outdir>`: starts `llama-server` from each build with an identical command line and runs three prompts (code, chat, ~32k filler) at `temp 0`, 3 repetitions each. It records tg/pp speed, `draft_n`/`draft_n_accepted`, output sha256 and an image smoke test. Env: `TAGS=base,new|new`, `ROUNDS`, `SPEC_N`, `NEW_ROOT`.
- `bisect_step.sh <commit>`: builds a commit in a separate worktree and runs `server_ab.py` on it. Optional env: `TWIN_MAX`, `PATCH_FILE`/`PATCH_SED`, `PATCH_CMD` (patch applied before the build).
- `compare_server.py <dir>`: compares `new_r1.json` with `base_r1.json`.
- `cmp_remote.sh <host result dir>`: copies a result dir from the host and compares it. Needs `AB_HOST=user@host`.
- `revert_hunks.py`, `hunk_bisect.sh`: revert selected hunks of a diff and rebuild/test each variant.
- `variants*.sh`, `pr234_*.sh`: knob, patch and PR variant sweeps.
- `bench_ab.sh`, `compare_bench.py`, `fattn_ab.sh`: `llama-bench` and FLASH_ATTN_EXT checks.

## Example

```sh
# on the host, as the user that owns the build directories
sudo -n -u lemonade bash -s -- <commit> < bisect_step.sh

# on your machine
AB_HOST=user@host bash cmp_remote.sh /tmp/srvbisect_<commit>_<timestamp>
```

## Caveats

- One round per variant by default. Acceptance counts are repeatable; tg varies by a few percent.
- Chat outputs are not deterministic even on the baseline, so their hashes are a weak signal.
- Make sure no other `llama-server` is using the GPU while a test runs.
