import os
import sys
import threading
import numpy as np
from langchain_core.embeddings import Embeddings

_model_instance = None
_model_lock = threading.Lock()

class ONNXEmbeddings(Embeddings):
    """Local, high-accuracy CPU embeddings model using onnxruntime and tokenizers."""
    def __init__(self, model_name: str):
        # Resolve HF snapshot folder path dynamically
        model_cache_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}/snapshots")
        
        self.use_api = False
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
                
        # 2. If files are missing from local disk, fallback to Hugging Face Inference API
        # (Avoids downloading 133MB on Vercel which hits Vercel's strict 10s timeout limit)
        if not onnx_path or not tokenizer_path:
            self.use_api = True
            # Translate local ONNX model name to standard Hugging Face repo path for the Inference API
            self.model_name = "BAAI/bge-small-en-v1.5"
            print("Using Hugging Face Inference API for embeddings (Serverless mode)")
            return
            
        import onnxruntime as ort
        from tokenizers import Tokenizer
            
        print(f"Loading ONNX session from: {onnx_path}")
        self.session = ort.InferenceSession(onnx_path)
        
        print(f"Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding(direction="right", pad_id=0, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=512)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self.use_api:
            import urllib.request
            import json
            url = f"https://api-inference.huggingface.co/models/{self.model_name}"
            payload = {"inputs": texts}
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            }
            
            hf_token = os.getenv("HF_API_TOKEN")
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"
                
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            try:
                with urllib.request.urlopen(req) as response:
                    res_data = response.read().decode("utf-8")
                    embeddings = json.loads(res_data)
                    if isinstance(embeddings, list) and len(embeddings) > 0:
                        if isinstance(embeddings[0], list):
                            return embeddings
                        elif isinstance(embeddings[0], float):
                            return [embeddings]
                    raise ValueError(f"Unexpected response format from HF Inference API: {embeddings}")
            except Exception as e:
                print(f"Error querying Hugging Face Inference API: {e}")
                # Fallback to zero vectors if API is unavailable or rate-limited
                return [[0.0] * 384 for _ in texts]

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
