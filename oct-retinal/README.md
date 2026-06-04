# OCT Retinal — GS Transform FL Benchmark

4-class retinal OCT classification (CNV / DME / DRUSEN / NORMAL)
in a 3-client federated learning simulation, comparing raw vs GS-transformed inputs.

## Results

Pre-run results from our experiments are in [`results/`](results/) (JSON per run)
and plots are in [`plots/`](plots/).

| Condition | AUC | F1 | vs Baseline |
|-----------|-----|----|-------------|
| baseline | 0.954 | 0.819 | — |
| gs_0 | 0.958 | 0.823 | +0.004 |
| gs_20 | 0.943 | 0.788 | -0.011 |
| gs_50 | 0.892 | 0.658 | -0.062 |

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
and your Kaggle credentials. It installs all Python dependencies and
the [GSTransformFL](https://github.com/seha-ay/GSTransformFL) package.

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
Downloads OCT2017 from Kaggle (~6GB), converts to float32 NPY,
and prints class counts. Expected: ~108,000 images across 4 classes.

**Step 5 — Review site distributions in config.py:**
```bash
python config.py
```
Verify `SITE_DISTRIBUTIONS` matches your class counts.

**Step 6 — Launch the full benchmark:**
```bash
nohup python -u runs/run.py > runs/benchmark.log 2>&1 &
echo "PID: $!"
```
Runs all 4 conditions sequentially (baseline → gs_50 → gs_20 → gs_0).
Data split runs automatically at the start.

**Step 7 — Monitor progress:**
```bash
python monitor.py
```

**Step 8 — Generate plots after all runs complete:**
```bash
python plots/compare.py
```

## Configuration

Key settings in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_ROUNDS` | 10 | FL rounds per condition |
| `LOCAL_EPOCHS` | 3 | Local epochs per round |
| `N_REPEATS` | 3 | Independent repeats |
| `GS_ITER_COUNT` | 50 | GS algorithm iterations |

## Dataset

- **Source:** [paultimothymooney/kermany2018](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)
- **Classes:** CNV=0, DME=1, DRUSEN=2, NORMAL=3
- **Size:** ~108,000 images across 4 classes
- **Split:** patient-aware, no patient overlap across sites or test set
