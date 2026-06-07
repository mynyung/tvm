#!/bin/bash
set -e

# 0. sudo 권한 유지 설정
sudo -v
while true; do
    sudo -n true
    sleep 60
done 2>/dev/null &
SUDO_KEEPALIVE_PID=$!

# 스크립트 종료 시(에러 포함) 정리 작업 예약
cleanup() {
    echo "Cleaning up..."
    sudo nvidia-smi -i $GPU_ID -rgc || true
    kill $SUDO_KEEPALIVE_PID 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

# 1. 환경 설정
export CUDA_VISIBLE_DEVICES=0
GPU_ID=0
MODELS=(rtdetr-r50 exaone-deep-7.8b qwen3.5-9b deepseek-r1-distill-14b)

FREQS=(1410 1710 2235 2385 2520)

# 2. Dataset Generation을 위한 빌드 (Power 측정 기능 활성화)
# 이미 해당 설정으로 빌드되어 있다면 이 과정은 매우 빠르게 넘어갑니다.
# echo "===================================="
# echo "START BUILD (NVML_POWER=ON)"
# echo "===================================="
# cd ~/tvm/build
# cmake .. -DTVM_ENABLE_NVML_POWER=ON
# make -j$(nproc)

# 3. Dataset Generation (전력 데이터 수집 전용)
# 튜닝 로그를 기반으로 실제 전력 소모량을 측정하여 데이터셋에 추가합니다.
echo "====================================" 
echo "START POWER DATASET COLLECTION"
echo "===================================="
cd ~/tvm/tutorials
sudo nvidia-smi -i $GPU_ID -pm 1

for MODEL in "${MODELS[@]}"
do
    echo "[INFO] Processing Model: $MODEL"
    for FREQ in "${FREQS[@]}"
    do
        echo ">>> Frequency Locking: $FREQ MHz"
        # GPU 클럭 고정
        sudo nvidia-smi -i $GPU_ID -lgc $FREQ,$FREQ
        sleep 3 # 전력 측정 안정화를 위한 하드웨어 대기

        # 기존 튜닝 로그를 읽어 전력 데이터를 수집 및 저장
        python append_dataset.py \
            --freq $FREQ \
            --model $MODEL \
            --max_samples 5000
    done
    echo "[SUCCESS] Model $MODEL Power Dataset Collection Done"
done

echo "ALL POWER DATASET EXPERIMENTS FINISHED SUCCESSFULLY"