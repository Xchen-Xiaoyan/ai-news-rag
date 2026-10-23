from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
INDEX = DATA / "index"

CORPUS_RAW = RAW / "corpus_135k.parquet"
HIGH_CONF_RAW = RAW / "high_conf.parquet"
DOCS = PROCESSED / "docs.parquet"
CHUNKS = PROCESSED / "chunks.parquet"

# Subset: articles the original relevance model scored as clearly about AI's industry impact,
# plus every article in the hand-validated high-confidence set.
MIN_FINAL_SCORE = 4

# Chunking (words, not tokens; ~1.3 tokens/word keeps chunks under the 512-token encoder limit)
CHUNK_TARGET_WORDS = 200
CHUNK_MAX_WORDS = 300
CHUNK_OVERLAP_SENTENCES = 1
MIN_CHUNK_WORDS = 40

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"
# bge models expect this instruction prefix on queries (not on passages)
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
