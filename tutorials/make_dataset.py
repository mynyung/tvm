import csv
import numpy as np
import tvm
from tvm import tir
from tvm import meta_schedule as ms
import hashlib

# ===============================
# Basic Setup
# ===============================

dev = tvm.cuda(0)
N = 100
OUT_PATH = "eyas_gpu_dataset.csv"

def as_float_metric(x):
    if hasattr(x, "ratio"):
        return float(x.ratio)
    return float(x)

def make_nd(shape, dtype):
    arr = np.random.rand(*[int(s) for s in shape]).astype(dtype)
    return tvm.nd.array(arr, device=dev)

def trace_fingerprint(trace):
    obj = trace.as_python() if hasattr(trace, "as_python") else repr(trace)
    s = obj if isinstance(obj, str) else "\n".join(str(x) for x in obj)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ===============================
# Load Tuning Records
# ===============================

db = ms.database.JSONDatabase(work_dir="tuning_logs")
all_recs = db.get_all_tuning_records()
actual_N = min(N, len(all_recs))
recs = [all_recs[i] for i in range(actual_N)]

print(f"[INFO] Loaded {len(recs)} tuning records.")

extractor = ms.feature_extractor.PerStoreFeature()

# ===============================
# Dataset Generation
# ===============================

with open(OUT_PATH, "w", newline="") as f:
    writer = csv.writer(f)

    header = [
        "i",
        "workload_hash",
        "trace_hash",
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
            ctx = ms.TuneContext(mod=mod, target=target, task_name=f"rec_{i}")

            # Feature extraction (n_store × 164)
            (feat_nd,) = extractor.extract_from(ctx, candidates=[cand])
            feat = feat_nd.numpy()

            # Eyas aggregation → 656
            agg = np.concatenate(
                [feat.mean(0), feat.std(0), feat.min(0), feat.max(0)]
            )

            # Build & run
            rt_mod = tvm.build(sch.mod, target=target)
            args = [make_nd(t.shape, t.dtype) for t in r.args_info]

            ftimer = rt_mod.time_evaluator(
                "main",
                dev,
                number=1,
                repeat=3,
                min_repeat_ms=50,
            )

            timing = ftimer(*args)

            # 🔥 Read NVML snapshot from C++ runtime
            nvml = tvm.get_global_func(
                "runtime.profiling.get_last_nvml_metrics"
            )()

            # avg_power = as_float_metric(nvml["avg_power_w"])
            avg_power = float(tvm.get_global_func("runtime.profiling.get_last_nvml_metrics")())

            row = [
                i,
                workload_hash,
                trace_hash,
                int(feat.shape[0]),
                float(timing.mean) * 1e3,
                avg_power,
            ]

            row += [float(x) for x in agg.tolist()]
            writer.writerow(row)

            print(
                f"[{i+1}/{actual_N}] "
                f"Lat: {float(timing.mean)*1e3:.3f} ms | "
                f"Power: {avg_power:.2f} W"
            )

        except Exception as e:
            print(f"[{i+1}/{actual_N}] Failed: {e}")

print("✅ Dataset written to:", OUT_PATH)