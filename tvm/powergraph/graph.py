import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tvm
from tvm import tir
from tvm import meta_schedule as ms
import argparse
import random

# ===============================
# 1. 설정 및 경로
# ===============================
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model name (e.g., resnet50)")
parser.add_argument("--num_recs", type=int, default=30, help="Number of schedules to profile")
args = parser.parse_args()

MODEL = args.model
NUM_RECS = args.num_recs
TUNING_LOG_DIR = f"/home/hyunjae/tvm/tutorials/tuning_logs_{MODEL}"
GRAPH_DIR = "/home/hyunjae/tvm/powergraph"

# 폴더가 없으면 생성
if not os.path.exists(GRAPH_DIR):
    os.makedirs(GRAPH_DIR)

dev = tvm.cuda(0)

# ===============================
# 2. 그래프 그리기 함수
# ===============================
def plot_latest_trace(csv_path):
    df = pd.read_csv(csv_path)
    # 샘플링 주기는 C++ 코드에서 5ms로 설정됨
    df['time_ms'] = df['sample_index'] * 5
    
    plt.figure(figsize=(12, 6))
    
    # 구간 구분 (-1: 웜업, 0+: 본 측정)
    warmup_df = df[df['iteration_id'] == -1]
    measure_df = df[df['iteration_id'] >= 0]
    
    if not warmup_df.empty:
        plt.plot(warmup_df['time_ms'], warmup_df['power_w'], color='red', linestyle='--', label='Warmup', alpha=0.6)
    
    if not measure_df.empty:
        # 반복 회차별로 색상을 다르게 표현 가능
        unique_iters = measure_df['iteration_id'].unique()
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_iters)))
        
        for idx, iter_id in enumerate(unique_iters):
            subset = measure_df[measure_df['iteration_id'] == iter_id]
            plt.plot(subset['time_ms'], subset['power_w'], color=colors[idx], label=f'Iteration {iter_id}', linewidth=2)

    plt.title(f"Power Profile: {os.path.basename(csv_path)}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Power (Watts)")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    out_png = csv_path.replace('.csv', '.png')
    plt.savefig(out_png)
    plt.close()
    print(f"  [Graph Saved] {os.path.basename(out_png)}")

# ===============================
# 3. 메인 실행 루프
# ===============================
def run_profiling():
    print(f"[*] Loading database from {TUNING_LOG_DIR}...")
    if not os.path.exists(TUNING_LOG_DIR):
        print(f"[!] Error: Tuning log directory not found.")
        return

    db = ms.database.JSONDatabase(work_dir=TUNING_LOG_DIR)
    all_recs = list(db.get_all_tuning_records())
    
    # 튜닝 로그에서 랜덤하게 NUM_RECS개 선택
    actual_sample_count = min(len(all_recs), NUM_RECS)
    recs = random.sample(all_recs, actual_sample_count)

    print(f"[*] Starting profiling for {actual_sample_count} random schedules (out of {len(all_recs)})...")

    for i, r in enumerate(recs):
        try:
            # 커널 빌드
            mod = r.workload.mod
            sch = tir.Schedule(mod)
            r.trace.apply_to_schedule(sch, remove_postproc=True)
            rt_mod = tvm.build(sch.mod, target=r.target)
            
            # 더미 입력 생성
            args = [tvm.nd.array(np.random.uniform(size=t.shape).astype(t.dtype), device=dev) for t in r.args_info]
            
            # Time Evaluator 호출 (Canvas에 구현된 C++ WrapTimeEvaluator가 실행됨)
            # 실행 직후 powergraph 폴더에 csv가 생성됨
            ftimer = rt_mod.time_evaluator("main", dev, number=50, repeat=3, min_repeat_ms=150)
            print(f"[{i+1}/{actual_sample_count}] Running Kernel...")
            ftimer(*args)

            # 가장 최근에 생성된 csv 파일 찾아서 그래프 그리기
            list_of_files = glob.glob(os.path.join(GRAPH_DIR, 'sched_*.csv'))
            if list_of_files:
                latest_csv = max(list_of_files, key=os.path.getctime)
                plot_latest_trace(latest_csv)

        except Exception as e:
            print(f"[{i+1}/{actual_sample_count}] Failed: {e}")

if __name__ == "__main__":
    run_profiling()
    print("[*] All tasks completed.")