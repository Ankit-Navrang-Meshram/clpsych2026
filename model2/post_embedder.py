# post_embedder.py
import torch
import re
import numpy as np
import spacy
import gensim
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer



# ── Twitter-RoBERTa task names ───────────────────────────────────────────────
_TASKS = ["emoji", "emotion", "hate", "irony", "offensive", "sentiment"]


class PostEmbedder:
    def __init__(self, wv_model_path: str, spacy_model: str = "en_core_web_sm", device: str = "cpu") -> None:
        print("[PostEmbedder] Loading Word2Vec …")
        self.wv_model = gensim.models.KeyedVectors.load_word2vec_format( wv_model_path, binary=False)
        self._wdim = self.wv_model["word"].shape[0]

        print("[PostEmbedder] Loading sentence-transformer …")
        self.sv_model = SentenceTransformer("sentence-transformers/nli-roberta-large", device=device)

        print("[PostEmbedder] Loading Twitter-RoBERTa task models …")
        self._task_models: dict[str, tuple] = {}
        for task in _TASKS:
            model_name = (f"cardiffnlp/twitter-roberta-base-{task}-latest" if task == "hate" else f"cardiffnlp/twitter-roberta-base-{task}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
            model.eval()
            self._task_models[task] = (model, tokenizer)

        print("[PostEmbedder] Loading spaCy …")
        self.nlp = spacy.load(spacy_model)

        self._stops = set(stopwords.words("english"))
        self._device = device
        print("[PostEmbedder] Ready.")


    def embed(self, text: str) -> np.ndarray:
        try:
            sentences = [str(s) for s in self.nlp(text).sents]
            if not sentences:
                sentences = [text]

            wv_part = self._word2vec_emb(text)
            sv_part = self._sentence_emb(sentences)
            task_part = self._task_scores(sentences)

            vec = np.concatenate([wv_part, sv_part, task_part], axis=None)

            if np.isnan(vec).any():
                raise ValueError(f"NaN values in embedding for text: {text[:60]!r}")
            return vec
        except Exception as exc:
            raise RuntimeError(f"[PostEmbedder] embed() failed: {exc}") from exc


    @staticmethod
    def _preprocess(text: str) -> str:
        """Lower-case; replace @mentions with @user; strip URLs."""
        tokens = []
        for t in text.split():
            t = t.lower()
            if t.startswith("@") and len(t) > 1:
                t = "@user"
            elif t.startswith("http"):
                t = ""
            tokens.append(t)
        return " ".join(tokens)

    def _remove_stopwords(self, text: str) -> list[str]:
        return [w for w in text.split() if w and w not in self._stops]

    def _word2vec_emb(self, text: str) -> np.ndarray:
        cleaned = self._preprocess(text)
        words = self._remove_stopwords(cleaned)
        vec = np.zeros(self._wdim)
        n = 0
        for w in words:
            if w in self.wv_model:
                vec += self.wv_model[w]
                n += 1
        if n > 0:
            vec /= n
        return vec

    def _sentence_emb(self, sentences: list[str]) -> np.ndarray:
        embeddings = self.sv_model.encode(sentences, device=self._device)
        return np.mean(embeddings, axis=0)

    def _task_score_single(self, task: str, text: str) -> np.ndarray:
        model, tokenizer = self._task_models[task]
        enc = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = model(**enc)
        return out[0][0].detach().cpu().numpy()

    def _task_scores(self, sentences: list[str]) -> np.ndarray:
        """Average Twitter-RoBERTa scores across all sentences (hate excluded from concat)."""
        per_sentence = []
        for sent in sentences:
            parts = [
                self._task_score_single(t, sent)
                for t in ["emoji", "emotion", "irony", "offensive", "sentiment" , "hate"]
            ]
            per_sentence.append(np.concatenate(parts, axis=None))
        return np.mean(per_sentence, axis=0)