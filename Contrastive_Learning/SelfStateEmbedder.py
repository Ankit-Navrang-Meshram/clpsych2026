

import torch
import numpy as np
from transformers import DistilBertTokenizerFast

from contrastive_model import SelfStateEncoder
from data_structure import Post
from typing import List



class SelfStateEmbedder:


    def __init__(self, model_path: str, embedding_dim: int = 128, device: str = "cpu", max_length: int = 128) -> None:
        self.device     = device
        self.max_length = max_length

        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.model     = SelfStateEncoder(embedding_dim=embedding_dim)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device).eval()

        print(
            f"[SelfStateEmbedder] Ready — model={model_path}, "
            f"dim={embedding_dim}, device={device}"
        )

    @torch.no_grad()
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string → (embedding_dim,) numpy array."""
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        emb = self.model.encode(enc["input_ids"], enc["attention_mask"])
        return emb.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of texts → (N, embedding_dim) numpy array."""
        chunks = []
        for start in range(0, len(texts), batch_size):
            enc = self.tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            emb = self.model.encode(enc["input_ids"], enc["attention_mask"])
            chunks.append(emb.cpu().numpy())
        return np.concatenate(chunks, axis=0)

    # def enrich_post(self, post: Post) -> np.ndarray:
    #     """
    #     Convenience method: embed post.text and concatenate with post.post_embedding.
    #     Returns the full combined vector, or just the state embedding if
    #     post.post_embedding is None.
    #     """
    #     state_emb = self.embed(post.text)
    #     if post.post_embedding is None:
    #         return state_emb
    #     return np.concatenate([post.post_embedding, state_emb])