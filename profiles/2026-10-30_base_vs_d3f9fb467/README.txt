rocprofv3 kernel profiles: baseline 6aa81b88e vs d3f9fb467 (master e38359599 + PR #234 v2), no patches
GPU: gfx1151 (Radeon 8060S, Ryzen AI MAX+ 395), ROCm 10.1
Model: Qwen3.8-Flash-Next UD-Q4_K_XL, no MTP (llama-bench only)

Command (per dir, LLAMA_MTP_RESIDENT=0):
  rocprofv3 --kernel-trace --stats --output-format csv -d <dir> -o run -- \
    llama-bench -m <model> -ngl 999 -fa 1 -b 4096 -ub 4096 -lzm on-direct -p <1|3> -n 0 -r 50 -o json

Dirs:
  pp1_base, pp1_new  - 1-token batches (decode)
  pp3_base, pp3_new  - 3-token batches (= MTP verify batch with --spec-draft-n-max 2)
Files:
  run_kernel_stats.csv  - per-kernel calls / total / avg ns
  run_kernel_trace.csv.gz - full timeline (gunzip) (Start/End_Timestamp per dispatch, includes model load and warmup)
  run_agent_info.csv    - device info
  llama-bench.json      - t/s of the profiled run

Key result (pp3, whole run incl. load): MMVQ Q8_0 (mul_mat_vec_q<Q8_0,3>) 24939 -> 9741 calls (-752 ms);
mmb_tq8_dense_kernel<128,64,...> + <128,256,...> 0 -> 15198 calls (+1344 ms); not present at pp1.
