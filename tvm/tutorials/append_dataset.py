import csv
import numpy as np
import tvm
from tvm import tir
from tvm import meta_schedule as ms
import hashlib
import os
import argparse
import pandas as pd
import random

# ===============================
# Arguments
# ===============================

parser = argparse.ArgumentParser()
parser.add_argument("--freq", type=int, required=True, help="GPU frequency is required")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--max_samples", type=int, default=100) 
args = parser.parse_args()

FREQ_MHZ = args.freq
MODEL = args.model
MAX_SAMPLES = args.max_samples # 이제 이 값이 5000이면 10000개 중 5000개만 사용함

# ===============================
# Paths
# ===============================

TUNING_LOG_DIR = f"tuning_logs_{MODEL}"
OUT_PATH = f"eyas_gpu4090_dataset_{MODEL}.csv"
SCHEDULE_CACHE = f"schedule_subset_{MODEL}.txt"

# ===============================
# Device
# ===============================

dev = tvm.cuda(0)

# ===============================
# Load tuning records
# ===============================

db = ms.database.JSONDatabase(work_dir=TUNING_LOG_DIR)
all_recs = list(db.get_all_tuning_records())

print("Total tuning records in DB:", len(all_recs))

# ===============================
# Fix schedule subset (10,000개 고정 풀 생성 및 로드)
# ===============================

if os.path.exists(SCHEDULE_CACHE):
    # 캐시가 있으면 무조건 로드 (기존에 10000개로 생성된 것)
    with open(SCHEDULE_CACHE) as f:
        idxs = [int(x.strip()) for x in f]
    recs = [all_recs[i] for i in idxs]
    print(f"Loaded fixed schedule subset (Total pool: {len(recs)})")
else:
    # 캐시가 없으면 처음 생성할 때 10,000개를 타겟으로 생성
    POOL_SIZE = 10000 
    actual_N = min(POOL_SIZE, len(all_recs))
    idxs = random.sample(range(len(all_recs)), actual_N)
    with open(SCHEDULE_CACHE, "w") as f:
        for i in idxs:
            f.write(f"{i}\n")
    recs = [all_recs[i] for i in idxs]
    print(f"Created new 10,000-sample pool and saved to {SCHEDULE_CACHE}")

# ===============================
# CSV setup (진행도 체크)
# ===============================

write_header = not os.path.exists(OUT_PATH)

if os.path.exists(OUT_PATH):
    df_existing = pd.read_csv(OUT_PATH)
    done_count = len(df_existing[df_existing['freq_mhz'] == FREQ_MHZ])
    start_i = done_count
else:
    start_i = 0

print(f"Frequency: {FREQ_MHZ} MHz | Progress: {start_i}/{MAX_SAMPLES}")

# ===============================
# Feature extractor & Helpers
# ===============================

extractor = ms.feature_extractor.PerStoreFeature()
tensor_cache = {}

def get_cached_nd(shape, dtype):
    key = (tuple(shape), str(dtype))
    if key not in tensor_cache:
        arr = np.random.rand(*[int(s) for s in shape]).astype(dtype)
        tensor_cache[key] = tvm.nd.array(arr, device=dev)
    return tensor_cache[key]

def trace_fingerprint(trace):
    obj = trace.as_python() if hasattr(trace, "as_python") else repr(trace)
    s = obj if isinstance(obj, str) else "\n".join(str(x) for x in obj)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ===============================
# Dataset generation
# ===============================

with open(OUT_PATH, "a", newline="") as f:
    writer = csv.writer(f)

    if write_header:
        header = ["i", "model", "workload_hash", "trace_hash", "freq_mhz", "n_stores", "lat_mean_ms", "avg_power_w"]
        header += [f"f{k}" for k in range(656)]
        writer.writerow(header)

    # 루프 시작 (start_i부터 MAX_SAMPLES까지)
    for i, r in enumerate(recs[start_i:], start=start_i):
        # [핵심] 10,000개의 pool 중 전달받은 MAX_SAMPLES(5000)번째에 도달하면 중단
        if i >= MAX_SAMPLES:
            print(f">>> Reached user-defined limit ({MAX_SAMPLES}). Stopping for {FREQ_MHZ}MHz.")
            break

        try:
            mod = r.workload.mod
            target = r.target
            workload_hash = int(tvm.ir.structural_hash(mod))
            trace_hash = trace_fingerprint(r.trace)

            sch = tir.Schedule(mod, debug_mask="all")
            r.trace.apply_to_schedule(sch, remove_postproc=True)
            cand = ms.MeasureCandidate(sch=sch, args_info=r.args_info)
            ctx = ms.TuneContext(mod=mod, target=target, task_name=f"{MODEL}_{i}")

            (feat_nd,) = extractor.extract_from(ctx, candidates=[cand])
            feat = feat_nd.numpy()
            agg = np.concatenate([feat.mean(0), feat.std(0), feat.min(0), feat.max(0)])

            rt_mod = tvm.build(sch.mod, target=target)
            args = [get_cached_nd(t.shape, t.dtype) for t in r.args_info]

            ftimer = rt_mod.time_evaluator("main", dev, number=5, repeat=3, min_repeat_ms=150)
            timing = ftimer(*args)

            avg_power = float(tvm.get_global_func("runtime.profiling.get_last_nvml_metrics")())

            row = [i, MODEL, workload_hash, trace_hash, FREQ_MHZ, int(feat.shape[0]), float(timing.mean) * 1e3, avg_power]
            row += [float(x) for x in agg.tolist()]
            writer.writerow(row)

            print(f"[{i+1}/{MAX_SAMPLES}] Freq {FREQ_MHZ} | Lat {float(timing.mean)*1e3:.3f} ms | Power {avg_power:.2f} W")

        except Exception as e:
            print(f"[{i+1}/{MAX_SAMPLES}] Failed:", e)

print(f"Job done for {FREQ_MHZ} MHz.")