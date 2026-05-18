# REED: Resource-Element Energy Difference for Noncoherent OTA-FL

This repository contains the PyTorch code used for the experiments in
**Resource-Element Energy Difference for Noncoherent Over-the-Air Federated Learning**.

The code implements FedAvg with three aggregation modes:

- `clean`: ideal noiseless FedAvg aggregation
- `ota_reed`: noncoherent resource-element energy difference (REED) aggregation
- `CSIT_SELECT`: coherent CSIT-based wireless aggregation baseline

The main experiments use MNIST and Fashion-MNIST with IID and Dirichlet client partitions.

## Repository contents

```text
REED/
├── compare_mnist_3modes.py      # Main MNIST/Fashion-MNIST comparison driver
├── main_mnist.py                # Small single-run demo
├── config.py                    # Simulation/radio/algorithm config dataclasses
├── Models/build_system.py       # Dataset, model, client, server construction
├── clients/                     # Client update logic
├── comm/                        # REED and coherent aggregation/channel code
├── data/                        # IID/Dirichlet/shard partition utilities
├── server/                      # Server aggregation/evaluation utilities
├── trainers/                    # PyTorch local trainer
├── utils/                       # Model adapter, clipping, checkpoint helpers
├── scripts/                     # Example reproduction commands
└── requirements.txt
```

Generated results are intentionally excluded from the repository. New runs write outputs to `runs/` by default, and `runs/` is ignored by Git except for `runs/.gitkeep`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The datasets are downloaded automatically by `torchvision` to `~/.torch`.

## Quick smoke test

```bash
python compare_mnist_3modes.py \
  --dataset mnist \
  --partition iid \
  --rounds 1 \
  --clients 5 \
  --sample 5 \
  --repeats 1 \
  --modes clean,reed \
  --chips 1 \
  --tag smoke
```

## Example paper-style runs

MNIST, IID, REED with one paired observation per coordinate:

```bash
python compare_mnist_3modes.py \
  --dataset mnist \
  --partition iid \
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
  --tag mnist-iid-M1
```

Fashion-MNIST, Dirichlet alpha=0.3, chip-diverse REED:

```bash
python compare_mnist_3modes.py \
  --dataset fashion \
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
  --chips 4 \
  --repeats 10 \
  --fixed_selections \
  --modes all \
  --tag fashion-dirichlet-a0p3-M4
```

You can also run the provided scripts:

```bash
bash scripts/run_mnist_iid_m1.sh
bash scripts/run_mnist_dirichlet_m1.sh
bash scripts/run_fashion_iid_m1.sh
bash scripts/run_fashion_dirichlet_m1_m2_m4.sh
```

## Notes

- The implementation runs on CPU by default.
- Use `--rounds` and `--repeats` with small values for debugging.
- The `--chips` argument controls the number of paired REED observations per scalar coordinate.
- `--modes all` runs clean FedAvg, REED, and the coherent CSIT baseline with matched client selections.
