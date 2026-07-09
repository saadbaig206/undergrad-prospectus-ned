import os
import sys
import threading
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from langchain_core.embeddings import Embeddings

_model_instance = None
_model_lock = threading.Lock()

class ONNXEmbeddings(Embeddings):
    """Local, high-accuracy CPU embeddings model using onnxruntime and tokenizers."""
    def __init__(self, model_name: str):
        # Resolve HF snapshot folder path dynamically
        model_cache_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}/snapshots")
        if not os.path.exists(model_cache_dir):
            raise RuntimeError(
                f"Model cache folder not found: {model_cache_dir}. "
                f"Please ensure you run 'huggingface-cli download {model_name}' first."
            )
        
        snapshots = os.listdir(model_cache_dir)
        if not snapshots:
            raise RuntimeError(f"No downloaded snapshots found in cache directory: {model_cache_dir}")
        
        snapshot_path = os.path.join(model_cache_dir, snapshots[0])
        onnx_path = os.path.join(snapshot_path, "onnx", "model.onnx")
        tokenizer_path = os.path.join(snapshot_path, "tokenizer.json")
        
        if not os.path.exists(onnx_path) or not os.path.exists(tokenizer_path):
            raise RuntimeError(
                f"Required model files missing in snapshot: {snapshot_path}. "
                f"Expected onnx/model.onnx and tokenizer.json."
            )
            
        print(f"Loading ONNX session from: {onnx_path}")
        self.session = ort.InferenceSession(onnx_path)
        
        print(f"Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding(direction="right", pad_id=0, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=512)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        # 1. Tokenize inputs
        encoded = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encoded], dtype=np.int64)
        
        # 2. Forward pass through ONNX model
        outputs = self.session.run(
            ["last_hidden_state"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }
        )
        last_hidden_state = outputs[0]  # shape: [batch_size, seq_len, 384]
        
        # 3. Mean Pooling
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        mean_embeddings = sum_embeddings / sum_mask
        
        # 4. L2 Normalization
        norms = np.linalg.norm(mean_embeddings, ord=2, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        normalized_embeddings = mean_embeddings / norms
        
        # Return as list of lists of floats
        return [list(map(float, emb)) for emb in normalized_embeddings]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Batch process large documents to avoid memory spikes
        batch_size = 32
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results.extend(self._embed(batch))
        return results

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

def get_embeddings_model():
    global _model_instance
    if _model_instance is None:
        with _model_lock:
            if _model_instance is None:
                # Load configuration
                from dotenv import load_dotenv
                load_dotenv()
                model_name = os.getenv("EMBEDDING_MODEL_NAME", "onnx-community/NoInstruct-small-Embedding-v0-ONNX")
                _model_instance = ONNXEmbeddings(model_name)
    return _model_instance

def embed_query(text: str) -> list[float]:
    return get_embeddings_model().embed_query(text)

def embed_documents(texts: list[str]) -> list[list[float]]:
    return get_embeddings_model().embed_documents(texts)
