#!/bin/bash
set -e
sudo -v

while true; do
    sudo -n true
    sleep 60
done 2>/dev/null &
SUDO_KEEPALIVE_PID=$!

export CUDA_VISIBLE_DEVICES=0

GPU_ID=0

MODELS=(densenet169 mobilenetv3)
FREQS=(885 1410 1710 2010 2520)

cd ~/tvm/build 
cmake .. -DTVM_ENABLE_NVML_POWER=OFF
make -j$(nproc)
cd ~/tvm/tutorials

# ----------------------------------------
#  E2E Tuning (frequency independent)
# ----------------------------------------

for MODEL in "${MODELS[@]}"
do
    echo "===================================="
    echo "START E2E TUNING FOR MODEL: $MODEL"
    echo "===================================="

    python e2e_opt_model.py \
        --model $MODEL

    echo "E2E TUNING DONE: $MODEL"
done


cd ~/tvm/build
cmake .. -DTVM_ENABLE_NVML_POWER=ON
make -j$(nrpoc)
cd ~/tvm/tutorials

# ----------------------------------------
#  Dataset Generation
# ----------------------------------------

sudo nvidia-smi -i $GPU_ID -pm 1

for MODEL in "${MODELS[@]}"
do
    echo "====================================" 
    echo "START DATASET FOR MODEL: $MODEL"
    echo "===================================="

    for FREQ in "${FREQS[@]}"
    do
        echo "Running frequency $FREQ MHz"

        sudo nvidia-smi -i $GPU_ID -lgc $FREQ,$FREQ

        sleep 3

        python append_dataset.py \
            --freq $FREQ \
            --model $MODEL
    done

    echo "MODEL $MODEL DONE"
done

sudo nvidia-smi -i $GPU_ID -rgc

kill $SUDO_KEEPALIVE_PID

echo "ALL EXPERIMENTS FINISHED"