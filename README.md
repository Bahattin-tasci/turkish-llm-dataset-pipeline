# turkish-llm-dataset-pipeline
End-to-end Turkish LLM pretraining dataset pipeline. Streams BellaTurca, Cosmos, and FineWeb2, applies exact (SHA-256+SQLite) and near-duplicate (MinHash LSH) filtering, and uploads cleaned shards to HuggingFace.
