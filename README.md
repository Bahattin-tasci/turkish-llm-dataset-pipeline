# Turkish LLM Dataset Pipeline

End-to-end pipeline for building a large-scale Turkish pretraining dataset. Streams text from BellaTurca, Cosmos, and FineWeb2, applies exact and near-duplicate filtering, and uploads cleaned shards to HuggingFace.

The resulting dataset is available at: [tascib/turkish-llm-dataset](https://huggingface.co/datasets/tascib/turkish-llm-dataset)

---

## Dataset Sources

| Source | HuggingFace ID | Description |
|--------|---------------|-------------|
| BellaTurca | `turkish-nlp-suite/BellaTurca` | Cleaned academic and web Turkish corpus |
| Cosmos | `ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0` | Yıldız Technical University Turkish web corpus |
| FineWeb2 | `altaidevorg/fineweb-2-turkish-categorized` | Common Crawl based Turkish web text |

---

## Pipeline Overview

Each record passes through the following steps:

1. **Text extraction** — supports `text`, `content`, `sentence` fields
2. **Length filter** — keeps records between 20 and 500,000 characters
3. **Exact deduplication** — SHA-256 hash stored in SQLite; duplicates are discarded
4. **Near-duplicate detection** — MinHash LSH (threshold=0.85); similar texts are discarded
5. **Shard upload** — every 50,000 clean records are uploaded to HuggingFace as a JSONL file

Output format per record:
```json
{"text": "..."}
```

---

## Requirements

Python 3.11 or higher is recommended.

Install dependencies:

```bash
pip install datasets huggingface_hub datasketch tqdm
```

---

## HuggingFace Token Setup

The pipeline uploads data to HuggingFace and requires a **Write** token.

**Step 1 — Get your token:**
1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **"New token"**
3. Select **Write** permission
4. Copy the token (starts with `hf_...`)

**Step 2 — Authenticate:**

```bash
pip install huggingface_hub
huggingface-cli login
```

Paste your token when prompted. It will be saved to `~/.cache/huggingface/token` and reused automatically.

Alternatively, set it as an environment variable:

```bash
# Linux / macOS
export HF_TOKEN=hf_your_token_here

# Windows (PowerShell)
$env:HF_TOKEN="hf_your_token_here"
```

---

## Configuration

Edit the following constants at the top of `turkish_dataset_pipeline.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_REPO` | `.../......` | Target HuggingFace dataset repository |
| `PRIVATE_REPO` | `True` | Whether to keep the repo private |
| `SHARD_SIZE` | `50_000` | Number of records per uploaded shard |
| `MIN_CHARS` | `20` | Minimum text length in characters |
| `MAX_CHARS` | `500_000` | Maximum text length in characters |
| `LSH_THRESHOLD` | `0.85` | Near-duplicate similarity threshold |

---

## Usage

```bash
python turkish_dataset_pipeline.py run_stream
```

The pipeline supports **resuming** from where it left off. If interrupted, it saves a checkpoint to `pipeline_progress.json`. On the next run, it automatically continues from the last completed shard.

---

## Running on an HPC Cluster (SLURM)

Create a job script `run_pipeline.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=turkish_dataset
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --partition=long_mdbf
#SBATCH --qos=long_mdbf
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module load python/3.11.9

cd /path/to/your/project

python -m venv venv
source venv/bin/activate

pip install datasets huggingface_hub datasketch tqdm -q

python turkish_dataset_pipeline.py run_stream
```

Submit the job:

```bash
sbatch run_pipeline.sh
```

Monitor progress:

```bash
# Check job status
squeue -u your_username

# Follow logs
tail -f output.log
```

---

## Using the Dataset

```python
from datasets import load_dataset

ds = load_dataset("tascib/turkish-llm-dataset", split="train")
```

If you have a local sample file:

```python
from datasets import load_dataset

ds = load_dataset(
    "json",
    data_files="/path/to/sample_10gb.jsonl",
    split="train"
)
```

---

## Checkpoint & Resume

After each uploaded shard, the pipeline saves its state to `pipeline_progress.json`:

```json
{
    "source": "FineWeb2",
    "shard_id": 1500,
    "records_processed": 75000000
}
```

If the job is interrupted (e.g. time limit on HPC), simply resubmit the job. The pipeline will detect the checkpoint and resume automatically.

To start fresh, delete `pipeline_progress.json` and `exact_hashes.sqlite`.

---

## Project Structure

```
turkish-llm-dataset-pipeline/
├── turkish_dataset_pipeline.py   # Main pipeline script
├── run_pipeline.sh               # SLURM job script
├── .gitignore
└── README.md
```

---

## License

MIT
