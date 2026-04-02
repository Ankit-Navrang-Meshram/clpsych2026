#dataset.py
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data_structure import Post, Timeline, SelfState

class PostIndex:
    def __init__(self, timelines: List[Timeline], exclude_same_timeline: bool = True, skip_no_embedding: bool = True) -> None:
        self.exclude_same_timeline = exclude_same_timeline

        # Collect all (timeline_id, Post) pairs that have a valid embedding
        self._entries: List[Tuple[str, Post]] = []

        for tl in timelines:
            for post in tl.posts:
                if post.post_embedding is None:
                    if not skip_no_embedding:
                        raise ValueError(
                            f"Post {post.post_id!r} has no embedding. "
                            "Run PostEmbedder first."
                        )
                    continue
                self._entries.append((tl.timeline_id, post))

        if not self._entries:
            raise ValueError("No posts with embeddings found in the provided timelines.")

        # Stack into (N, D) float32 matrix and L2-normalise rows for cosine sim
        raw = np.stack([e[1].post_embedding for e in self._entries], axis=0).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)   # avoid /0 for zero vectors
        self._matrix = raw / norms                  # shape (N, D), unit vectors

        self._timeline_ids = [e[0] for e in self._entries]
        self._posts        = [e[1] for e in self._entries]

        print( f"[PostIndex] Built index: {len(self._entries)} posts, "f"embedding dim={raw.shape[1]}")

    def query( self, post: Post, query_timeline_id: str, k: int) -> Tuple[List[Post], List[float]]:
        if post.post_embedding is None:
            return [], []

        # Normalise query vector
        q = post.post_embedding.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        # Cosine similarities: dot(matrix, q) because rows are already normalised
        sims = self._matrix @ q  # shape (N,)

        # Mask out: (a) the query post itself, (b) same-timeline posts if requested
        for i, (tid, p) in enumerate(self._entries):
            if p.post_id == post.post_id:
                sims[i] = -2.0   # guaranteed lowest
            elif self.exclude_same_timeline and tid == query_timeline_id:
                sims[i] = -2.0

        # Top-k indices (descending)
        top_k_idx = np.argpartition(sims, -k)[-k:]          # unsorted
        top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]  # sorted desc

        similar_posts = [self._posts[i] for i in top_k_idx]
        scores        = [float(sims[i])  for i in top_k_idx]

        return similar_posts, scores

@dataclass
class TopKSimilarInstance:
    timeline_id:   str
    post:          Post
    similar_posts: List[Post]        # length == k (or fewer at dataset edges)
    scores:        List[float]       # parallel to similar_posts
    context_posts: List[Post]

    # ── convenience pass-throughs so code that reads Task11Instance still works
    @property
    def post_id(self)        -> str:           return self.post.post_id
    @property
    def post_index(self)     -> int:           return self.post.post_index
    @property
    def text(self)           -> str:           return self.post.text
    @property
    def adaptive_state(self) -> SelfState:     return self.post.adaptive_state
    @property
    def maladaptive_state(self) -> SelfState:  return self.post.maladaptive_state
    @property
    def wellbeing(self)      -> Optional[int]: return self.post.wellbeing
    @property
    def post_embedding(self) -> Optional[np.ndarray]: return self.post.post_embedding

    def similar_texts(self) -> List[str]:
        """Convenience: just the text of each similar post."""
        return [p.text for p in self.similar_posts]


# ── 3. Dataset ────────────────────────────────────────────────────────────────

class TopKSimilarDataset(Dataset):
    def __init__(self, timelines: List[Timeline], index: PostIndex, k: int = 5, t: int = 3, annotated_only: bool = True) -> None:
        self.k = k
        self.t = t
        self.instances: List[TopKSimilarInstance] = []

        skipped = 0
        for tl in timelines:
            for i, post in enumerate(tl.posts):
                if annotated_only and not post.is_annotated:
                    continue
                if post.post_embedding is None:
                    skipped += 1
                    continue

                similar_posts, scores = index.query(post, tl.timeline_id, k=k)
                context_posts = tl.get_context(i, window=t)

                self.instances.append(TopKSimilarInstance(
                    timeline_id=tl.timeline_id,
                    post=post,
                    similar_posts=similar_posts,
                    scores=scores,
                    context_posts=context_posts
                ))

        print(
            f"[TopKSimilarDataset] {len(self.instances)} instances built "
            f"(k={k}, skipped {skipped} posts without embedding)"
        )

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, i: int) -> TopKSimilarInstance:
        return self.instances[i]

    # ── collate ───────────────────────────────────────────────────────────────

    @staticmethod
    def collate(batch: List[TopKSimilarInstance]) -> dict:
        posts         = [inst.post          for inst in batch]
        similar_posts = [inst.similar_posts for inst in batch]
        context_posts = [inst.context_posts for inst in batch]
        scores        = torch.tensor([inst.scores for inst in batch], dtype=torch.float32)


        # Labels
        ada_presence = torch.tensor([inst.post.adaptive_state.presence for inst in batch], dtype=torch.float32)
        mal_presence = torch.tensor([inst.post.maladaptive_state.presence for inst in batch], dtype=torch.float32)

        # 1. Target Post Embeddings: (B, D)
        embs = [inst.post.post_embedding for inst in batch]
        post_embeddings = torch.tensor(np.stack(embs), dtype=torch.float32) if all(e is not None for e in embs) else None

        # 2. Stack similar-post embeddings: shape (B, k, D)
        sim_embs = [[p.post_embedding for p in inst.similar_posts] for inst in batch]
        if all(e is not None for row in sim_embs for e in row):
            similar_embeddings = torch.tensor(
                np.stack([np.stack(row) for row in sim_embs]), dtype=torch.float32
            )
        else:
            similar_embeddings = None

        # 3. Temporal Context Embeddings: (B, T, D) 
        # Note: We must pad with zeros if a timeline has fewer than 't' previous posts
        ctx_embs_list = []
        max_t = max(len(inst.context_posts) for inst in batch) if batch else 0
        
        # Determine embedding dimension from first available embedding
        dim = embs[0].shape[0] if embs[0] is not None else 0

        for inst in batch:
            # Get existing embeddings
            current_ctx = [p.post_embedding for p in inst.context_posts]
            # Pad with zero vectors if the user is at the start of their timeline
            while len(current_ctx) < max_t:
                current_ctx.insert(0, np.zeros(dim)) 
            ctx_embs_list.append(np.stack(current_ctx))

        context_embeddings = torch.tensor(np.stack(ctx_embs_list), dtype=torch.float32) if dim > 0 else None
        
        return {
            "posts":              posts,
            "similar_posts":      similar_posts,
            "context_posts":      context_posts,      # List of lists
            "scores":             scores,             
            "post_embeddings":    post_embeddings,    
            "similar_embeddings": similar_embeddings, 
            "context_embeddings": context_embeddings, # (B, T, D) tensor
            "ada_presence":       ada_presence,       
            "mal_presence":       mal_presence,       
        }