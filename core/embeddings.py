import threading
from langchain_core.embeddings import Embeddings

_model_instance = None
_model_lock = threading.Lock()

class Model2VecEmbeddings(Embeddings):
    """Local, CPU-friendly embeddings model wrapper using model2vec."""
    def __init__(self, model_name: str):
        from model2vec import StaticModel
        # This will download the model locally on first run
        self.model = StaticModel.from_pretrained(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # model2vec encode returns numpy arrays; convert to list of floats for serialization
        embeddings = self.model.encode(texts)
        return [list(map(float, emb)) for emb in embeddings]

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode([text])[0]
        return list(map(float, embedding))

def get_embeddings_model():
    global _model_instance
    if _model_instance is None:
        with _model_lock:
            if _model_instance is None:
                _model_instance = Model2VecEmbeddings("minishlab/potion-base-8M")
    return _model_instance

def embed_query(text: str) -> list[float]:
    return get_embeddings_model().embed_query(text)

def embed_documents(texts: list[str]) -> list[list[float]]:
    return get_embeddings_model().embed_documents(texts)
