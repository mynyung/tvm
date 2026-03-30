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
parser.add_argument("--max_samples", type=int, default=100) #max power samples 개수. 8000정도가 좋을 듯.
args = parser.parse_args()

FREQ_MHZ = args.freq
MODEL = args.model
MAX_SAMPLES = args.max_samples

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

print("Total tuning records:", len(all_recs))

# ===============================
# Fix schedule subset (important)
# ===============================

if os.path.exists(SCHEDULE_CACHE):

    with open(SCHEDULE_CACHE) as f:
        idxs = [int(x.strip()) for x in f]

    recs = [all_recs[i] for i in idxs]

    print("Loaded fixed schedule subset")

else:

    actual_N = min(MAX_SAMPLES, len(all_recs))

    idxs = random.sample(range(len(all_recs)), actual_N)

    with open(SCHEDULE_CACHE, "w") as f:
        for i in idxs:
            f.write(f"{i}\n")

    recs = [all_recs[i] for i in idxs]

    print("Created schedule subset")

actual_N = len(recs)

print("Schedule count:", actual_N)
print("Frequency:", FREQ_MHZ)

# ===============================
# Feature extractor
# ===============================

extractor = ms.feature_extractor.PerStoreFeature()

# ===============================
# Tensor cache
# ===============================

tensor_cache = {}

def get_cached_nd(shape, dtype):

    key = (tuple(shape), str(dtype))

    if key not in tensor_cache:

        arr = np.random.rand(*[int(s) for s in shape]).astype(dtype)

        tensor_cache[key] = tvm.nd.array(arr, device=dev)

    return tensor_cache[key]

# ===============================
# Trace fingerprint
# ===============================

def trace_fingerprint(trace):

    obj = trace.as_python() if hasattr(trace, "as_python") else repr(trace)

    s = obj if isinstance(obj, str) else "\n".join(str(x) for x in obj)

    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ===============================
# CSV setup
# ===============================

write_header = not os.path.exists(OUT_PATH)

if os.path.exists(OUT_PATH):

    df = pd.read_csv(OUT_PATH)
    start_i = len(df)

else:

    start_i = 0

print("Start index:", start_i)

# ===============================
# Dataset generation
# ===============================

with open(OUT_PATH, "a", newline="") as f:

    writer = csv.writer(f)

    if write_header:

        header = [
            "i",
            "model",
            "workload_hash",
            "trace_hash",
            "freq_mhz",
            "n_stores",
            "lat_mean_ms",
            "avg_power_w",
        ]

        header += [f"f{k}" for k in range(656)]

        writer.writerow(header)

    for i, r in enumerate(recs):

        try:

            mod = r.workload.mod
            target = r.target

            workload_hash = int(tvm.ir.structural_hash(mod))
            trace_hash = trace_fingerprint(r.trace)

            # Apply schedule
            sch = tir.Schedule(mod, debug_mask="all")
            r.trace.apply_to_schedule(sch, remove_postproc=True)

            cand = ms.MeasureCandidate(sch=sch, args_info=r.args_info)

            ctx = ms.TuneContext(
                mod=mod,
                target=target,
                task_name=f"{MODEL}_{i}",
            )

            # Feature extraction
            (feat_nd,) = extractor.extract_from(ctx, candidates=[cand])
            feat = feat_nd.numpy()

            # Eyas aggregation
            agg = np.concatenate(
                [
                    feat.mean(0),
                    feat.std(0),
                    feat.min(0),
                    feat.max(0),
                ]
            )

            # Build kernel
            rt_mod = tvm.build(sch.mod, target=target)

            args = [get_cached_nd(t.shape, t.dtype) for t in r.args_info]

            # Latency measurement
            ftimer = rt_mod.time_evaluator(
                "main",
                dev,
                number=50,
                repeat=3,
                min_repeat_ms=150,
            )

            timing = ftimer(*args)

            # NVML power
            avg_power = float(
                tvm.get_global_func(
                    "runtime.profiling.get_last_nvml_metrics"
                )()
            )

            row = [
                start_i + i,
                MODEL,
                workload_hash,
                trace_hash,
                FREQ_MHZ,
                int(feat.shape[0]),
                float(timing.mean) * 1e3,
                avg_power,
            ]

            row += [float(x) for x in agg.tolist()]

            writer.writerow(row)

            print(
                f"[{i+1}/{actual_N}] "
                f"Lat {float(timing.mean)*1e3:.3f} ms "
                f"| Power {avg_power:.2f} W"
            )

        except Exception as e:

            print(f"[{i+1}/{actual_N}] Failed:", e)

print("Dataset appended to:", OUT_PATH)