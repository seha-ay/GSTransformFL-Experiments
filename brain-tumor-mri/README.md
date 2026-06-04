# Brain Tumor MRI — GS Transform FL Benchmark

4-class brain tumor classification (glioma / meningioma / notumor / pituitary)
in a 3-client federated learning simulation, comparing raw vs GS-transformed inputs.

## Results

Pre-run results from our experiments are in [`results/`](results/) (JSON per run)
and plots are in [`plots/`](plots/).

## Setup

**Step 1 — Clone both repos:**
```bash
git clone https://github.com/seha-ay/GSTransformFL-Experiments.git
cd GSTransformFL-Experiments/brain-tumor-mri
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
Downloads Brain Tumor MRI from Kaggle (~157MB), converts to float32 NPY,
and prints class counts. Expected output: 7,200 images, 1,800 per class.

**Step 5 — Review site distributions in config.py:**
```bash
python config.py
```
Verify `SITE_DISTRIBUTIONS` matches your class counts.
Default is balanced: 500 images per class per site, 150 per class for test.

**Step 6 — Launch the full benchmark:**
```bash
nohup python -u runs/run.py > runs/benchmark.log 2>&1 &
echo "PID: $!"
```
This runs all 4 conditions sequentially (baseline → gs_50 → gs_20 → gs_0).
Data split runs automatically at the start.

**Step 7 — Monitor progress:**
```bash
python monitor.py
```

**Step 8 — Generate plots after all runs complete:**
```bash
python plots/compare.py
```
Saves all comparison plots to `plots/`.

## Configuration

Key settings in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_ROUNDS` | 10 | FL rounds per condition |
| `LOCAL_EPOCHS` | 3 | Local epochs per round |
| `N_REPEATS` | 3 | Independent repeats |
| `GS_ITER_COUNT` | 50 | GS algorithm iterations |

To run a quick smoke test before committing to full run,
set `N_ROUNDS=3`, `LOCAL_EPOCHS=1`, `N_REPEATS=1` in `config.py`.

## Dataset

- **Source:** [masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Classes:** glioma=0, meningioma=1, notumor=2, pituitary=3
- **Size:** 7,200 images (1,800 per class)
- **Split:** stratified random, fixed seed per repeat
