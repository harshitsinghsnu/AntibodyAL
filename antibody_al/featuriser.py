"""
explainable_al/ab_featuriser.py
────────────────────────────────
ESM-2 featuriser for concatenated VH+G+VL sequences.
Replaces SMILES→ECFP for the Ab-Ag active learning task.

Design decisions
────────────────
- Single ESM-2 forward pass on VH+G+VL (simpler than dual encoder)
- Mean-pool ALL token hidden states (last layer) → [d] embedding
- No CDR identification needed — full VH+VL context captures affinity signal
- HDF5 cache: each sequence embedded once, reused across all experiments
- Batch inference: 32 seqs per forward pass → ~10x faster than one-at-a-time
"""
import hashlib
import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

ESM2_HF = {
    "8m":   "facebook/esm2_t6_8M_UR50D",
    "35m":  "facebook/esm2_t12_35M_UR50D",
    "150m": "facebook/esm2_t30_150M_UR50D",
    "650m": "facebook/esm2_t33_650M_UR50D",
}
ESM2_FAIR = {"150m": "esm2_t30_150M_UR50D", "650m": "esm2_t33_650M_UR50D"}
ESM2_DIM  = {"8m": 320, "35m": 480, "150m": 640, "650m": 1280}


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


class ESM2AbFeaturiser:
    def __init__(
        self,
        model_size:  str = "650m",
        device:      str = "auto",
        batch_size:  int = 32,
        cache_path:  str = "cache/ab_embeddings.h5",
        pool_mode:   str = "mean",
        layer:       int = -1,
    ):
        self.model_size = model_size
        self.device_str = device
        self.batch_size = batch_size
        self.pool_mode  = pool_mode
        self.layer      = layer
        self.dim        = ESM2_DIM[model_size]
        self._model     = None
        self._tok       = None
        self._backend   = None

        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self._cache_path = cache_path
        with h5py.File(cache_path, "a"):
            pass

    @property
    def device(self) -> torch.device:
        if self.device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_str)

    def _load(self):
        if self._model is not None:
            return
        logger.info(f"Loading ESM-2 {self.model_size} on {self.device} …")
        try:
            import esm as fair_esm
            model, alphabet = fair_esm.pretrained.load_model_and_alphabet(
                ESM2_FAIR[self.model_size])
            model.eval().to(self.device)
            self._tok      = alphabet.get_batch_converter()
            self._model    = model
            self._backend  = "fair_esm"
            self._n_layers = model.num_layers
            logger.info(f"  Ready via fair-esm")
            return
        except Exception as e:
            logger.info(f"  fair-esm failed ({e}), trying transformers …")

        from transformers import EsmModel, EsmTokenizer
        name = ESM2_HF[self.model_size]
        self._tok   = EsmTokenizer.from_pretrained(name)
        self._model = EsmModel.from_pretrained(
            name, output_hidden_states=True, device_map=None)
        self._model.eval().to(self.device)
        self._backend = "transformers"
        logger.info(f"  Ready via transformers")

    def _forward_batch(self, seqs: list[str]) -> list[np.ndarray]:
        self._load()
        if self._backend == "fair_esm":
            layer_idx = self._n_layers + self.layer if self.layer < 0 else self.layer
            labeled   = [(str(i), s) for i, s in enumerate(seqs)]
            _, _, tokens = self._tok(labeled)
            tokens = tokens.to(self.device)
            with torch.no_grad():
                out = self._model(tokens, repr_layers=[layer_idx],
                                  return_contacts=False)
            all_h = out["representations"][layer_idx].cpu().float()
            return [all_h[i, 1:len(s)+1, :].numpy() for i, s in enumerate(seqs)]
        else:
            inputs = self._tok(seqs, return_tensors="pt", padding=True,
                               truncation=True, max_length=1026,
                               add_special_tokens=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model(**inputs)
            all_h = out.hidden_states[self.layer].cpu().float()
            return [all_h[i, 1:len(s)+1, :].numpy() for i, s in enumerate(seqs)]

    def _cache_get(self, seq: str) -> Optional[np.ndarray]:
        key = f"esm2_{self.model_size}/{_hash(seq)}"
        try:
            with h5py.File(self._cache_path, "r") as f:
                return f[key][:] if key in f else None
        except Exception:
            return None

    def _cache_put(self, seq: str, emb: np.ndarray):
        key = f"esm2_{self.model_size}/{_hash(seq)}"
        try:
            with h5py.File(self._cache_path, "a") as f:
                if key not in f:
                    f.create_dataset(key, data=emb, compression="gzip")
        except Exception:
            pass

    def transform(self, sequences: list[str], show_progress: bool = True) -> np.ndarray:
        n = len(sequences)
        result, miss = [None]*n, []
        for i, seq in enumerate(sequences):
            c = self._cache_get(seq)
            if c is not None: result[i] = c
            else:             miss.append(i)

        bs = self.batch_size
        it = range(0, len(miss), bs)
        if show_progress and miss:
            it = tqdm(it, desc=f"ESM-2 {self.model_size}",
                      total=-(-len(miss)//bs))

        for start in it:
            batch_idx  = miss[start: start + bs]
            batch_seqs = [sequences[i][:1024] for i in batch_idx]
            batch_toks = self._forward_batch(batch_seqs)
            for j, gi in enumerate(batch_idx):
                tok = batch_toks[j]
                emb = (tok.mean(0) if self.pool_mode == "mean"
                       else tok.max(0)).astype(np.float32)
                result[gi] = emb
                self._cache_put(sequences[gi], emb)

        return np.stack(result).astype(np.float32)
