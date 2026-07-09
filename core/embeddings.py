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
        
        onnx_path = None
        tokenizer_path = None
        
        # 1. Check if files exist in local HuggingFace cache
        if os.path.exists(model_cache_dir):
            try:
                snapshots = os.listdir(model_cache_dir)
                if snapshots:
                    snapshot_path = os.path.join(model_cache_dir, snapshots[0])
                    cand_onnx = os.path.join(snapshot_path, "onnx", "model.onnx")
                    cand_tok = os.path.join(snapshot_path, "tokenizer.json")
                    if os.path.exists(cand_onnx) and os.path.exists(cand_tok):
                        onnx_path = cand_onnx
                        tokenizer_path = cand_tok
            except Exception as e:
                print(f"HF cache reading warning: {e}")
                
        # 2. Download files dynamically if cache is missing (e.g., on Vercel)
        if not onnx_path or not tokenizer_path:
            import urllib.request
            local_dir = "/tmp/model_cache"
            os.makedirs(local_dir, exist_ok=True)
            
            onnx_path = os.path.join(local_dir, "model.onnx")
            tokenizer_path = os.path.join(local_dir, "tokenizer.json")
            
            def download_with_user_agent(url: str, dest: str):
                print(f"Downloading {url} to {dest}...")
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                )
                with urllib.request.urlopen(req) as response, open(dest, "wb") as out_file:
                    out_file.write(response.read())
            
            if not os.path.exists(onnx_path):
                onnx_url = f"https://huggingface.co/{model_name}/resolve/main/onnx/model.onnx"
                try:
                    download_with_user_agent(onnx_url, onnx_path)
                except Exception as ex:
                    raise RuntimeError(f"Failed to download ONNX model from {onnx_url}: {ex}")
                    
            if not os.path.exists(tokenizer_path):
                tokenizer_url = f"https://huggingface.co/{model_name}/resolve/main/tokenizer.json"
                try:
                    download_with_user_agent(tokenizer_url, tokenizer_path)
                except Exception as ex:
                    raise RuntimeError(f"Failed to download tokenizer from {tokenizer_url}: {ex}")
            
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
