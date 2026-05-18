#!/usr/bin/env bash
set -euo pipefail
python compare_mnist_3modes.py \
  --dataset mnist \
  --partition dirichlet \
  --alpha 0.3 \
  --rounds 100 \
  --clients 10 \
  --sample 10 \
  --local_epochs 1 \
  --batch_size 64 \
  --local_lr0 0.05 \
  --local_lr_alpha 0.5 \
  --snr_db -10 \
  --chips 1 \
  --repeats 10 \
  --fixed_selections \
  --modes all \
  --tag mnist-dirichlet-a0p3-M1
