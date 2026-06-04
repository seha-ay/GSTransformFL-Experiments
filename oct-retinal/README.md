# OCT Retinal — GS Transform FL Benchmark

4-class retinal OCT classification (CNV / DME / DRUSEN / NORMAL)
in a 3-client federated learning simulation, comparing raw vs GS-transformed inputs.

## Results

Pre-run results from our experiments are in [`results/`](results/) (JSON per run)
and plots are in [`plots/`](plots/).

## Setup

**Step 1 — Clone the repo:**
```bash
git clone https://github.com/seha-ay/GSTransformFL-Experiments.git
cd GSTransformFL-Experiments/oct-retinal
```

**Step 2 — Install dependencies:**
```bash
bash setup.sh
```
This will prompt for your GitHub token (to install the gs_1ch package)
and your Kaggle credentials.

**Step 3 — Set up credentials:**
```bash
cp .env.template .env
# Edit .env and fill in your Kaggle username and API key
source .env
```

## Running the Experiment

**Step 4 — Download and convert the dataset:**
```bash
python data/convert.py
```
Downloads OCT2017 from Kaggle (~6GB), converts to float32 NPY.

**Step 5 — Launch the full benchmark:**
```bash
nohup python -u runs/run.py > runs/benchmark.log 2>&1 &
echo "PID: $!"
```

**Step 6 — Monitor progress:**
```bash
python monitor.py
```

**Step 7 — Generate plots after all runs complete:**
```bash
python plots/compare.py
```

## Dataset

- **Source:** [paultimothymooney/kermany2018](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)
- **Classes:** CNV=0, DME=1, DRUSEN=2, NORMAL=3
- **Split:** patient-aware, no patient overlap across sites or test set
