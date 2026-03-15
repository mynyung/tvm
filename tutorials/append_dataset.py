import csv
import numpy as np
import tvm
from tvm import tir
from tvm import meta_schedule as ms
import hashlib
import random
import os
import argparse
import pandas as pd

# ===============================
# Argument
# ===============================

parser = argparse.ArgumentParser()
parser.add_argument("--freq", type=int, required=True)
args = parser.parse_args()

FREQ_MHZ = args.freq

# ===============================
# Basic Setup
# ===============================

dev = tvm.cuda(0)

db = ms.database.JSONDatabase(work_dir="tuning_logs")
all_recs = list(db.get_all_tuning_records())

print("Total tuning records:", len(all_recs))

MAX_SAMPLES = 3000
OUT_PATH = "eyas_gpu4090_dataset_resnet50.csv"

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
# Load Records
# ===============================

actual_N = min(MAX_SAMPLES, len(all_recs))
recs = random.sample(all_recs, actual_N)

print(f"[INFO] Sampling {actual_N} records")

extractor = ms.feature_extractor.PerStoreFeature()

# ===============================
# CSV Handling
# ===============================

write_header = not os.path.exists(OUT_PATH)

if os.path.exists(OUT_PATH):
    df = pd.read_csv(OUT_PATH)
    start_i = len(df)
else:
    start_i = 0

print("Start index:", start_i)
print("Frequency:", FREQ_MHZ)

# ===============================
# Dataset Generation
# ===============================

with open(OUT_PATH, "a", newline="") as f:

    writer = csv.writer(f)

    if write_header:

        header = [
            "i",
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
                task_name=f"rec_{i}",
            )

            # Feature extraction
            (feat_nd,) = extractor.extract_from(ctx, candidates=[cand])

            feat = feat_nd.numpy()

            # Eyas aggregation (164 -> 656)
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

            # Run timing
            ftimer = rt_mod.time_evaluator(
                "main",
                dev,
                number=100,
                repeat=5,
                min_repeat_ms=300,
            )

            timing = ftimer(*args)

            # Power measurement
            avg_power = float(
                tvm.get_global_func(
                    "runtime.profiling.get_last_nvml_metrics"
                )()
            )

            row = [
                start_i + i,
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