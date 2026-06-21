import os


originalDataPath = "data/processed/Data.csv"
curatedDataPath = "data/processed/dataCurated.csv"

originalEmbeddingPath = "data/processed/embeddings.npy"
curatedEmbeddingPath = "data/processed/embeddingsCurated.npy"

originalIndexPath = "data/processed/data.index"
curatedIndexPath = "data/processed/dataCurated.index"


def getDataPath():
    return os.getenv(
        "CURATOR_DATA_PATH",
        curatedDataPath if os.path.exists(curatedDataPath) else originalDataPath,
    )


def getEmbeddingPath():
    return os.getenv(
        "CURATOR_EMBEDDING_PATH",
        curatedEmbeddingPath
        if os.path.exists(curatedEmbeddingPath)
        else originalEmbeddingPath,
    )


def getIndexPath():
    return os.getenv(
        "CURATOR_INDEX_PATH",
        curatedIndexPath if os.path.exists(curatedIndexPath) else originalIndexPath,
    )
