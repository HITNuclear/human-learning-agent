# Retrieval Benchmark (Resume-Friendly)

This folder provides a reproducible retrieval benchmark setup using common public datasets.

## Datasets

- Default: `beir/scifact/test` (common in resumes/interviews)
- Chinese option: `miracl/zh/dev`

Both are loaded through `ir_datasets`.

## Metrics

The script computes:

- `recall@5`
- `recall@10`
- `mrr@10`
- `ndcg@10`

## Install

From `backend/`:

```bash
pip install -r requirements.txt
pip install -r requirements-eval.txt
```

## Run (quick local baseline)

From `backend/`:

```bash
python -m eval.run_retrieval_benchmark \
  --dataset beir/scifact/test \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --top-k 10 \
  --max-docs 15000 \
  --max-queries 500 \
  --reset-index
```

## Run (Chinese)

```bash
python -m eval.run_retrieval_benchmark \
  --dataset miracl/zh/dev \
  --embedding-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --top-k 10 \
  --max-docs 20000 \
  --max-queries 500 \
  --reset-index
```

## Output

Results are written to:

- `backend/eval/results/metrics_<dataset>.json`

Example fields:

```json
{
  "dataset": "beir/scifact/test",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "top_k": 10,
  "indexed_docs": 15000,
  "evaluated_queries": 500,
  "recall@5": 0.4623,
  "recall@10": 0.5318,
  "mrr@10": 0.3844,
  "ndcg@10": 0.4179
}
```

## Notes

- For quick iteration, keep `--max-docs` and `--max-queries` limited.
- For final resume numbers, run with larger limits and fixed random-free settings.
- Keep model, top-k, and dataset fixed when comparing strategies.
