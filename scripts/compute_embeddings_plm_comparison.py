"""
Compute S0-style (VH+G+VL concatenation) embeddings for all 9 AbBiBench
datasets using 4 alternative protein/antibody language models, for a
same-architecture comparison against the ESM-2 650M results already in
data/embeddings/.

Models:
  - ProtBert    (Rostlab/prot_bert)         - general protein LM, BERT-style
  - AbLang2     (ablang2-paired)            - antibody-specific, paired VH|VL
  - AntiBERTy   (igfold's AntiBERTy)        - antibody-specific, tiny BERT
  - ProGen2     (hugohrban/progen2-small)   - general protein LM, GPT-style

ESM-C was attempted but its HF tokenizer integration is incompatible with
the installed transformers version in this environment (custom vocab format
not readable by EsmTokenizer/AutoTokenizer) - skipped, documented in REPORT.md.

Output layout matches data/embeddings/{dataset}/:
  data/embeddings_{plm}/{dataset}/X.npy, y.npy, top2p.npy, top5p.npy
(top2p/top5p/y are identical to the ESM-2 run - copied directly - only X differs)
"""
import os
os.environ.pop("SSL_CERT_FILE", None)

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

PROCESSED = Path(r"D:\explainable_AL\data\processed")
ESM2_EMB  = Path(r"D:\explainable_AL\data\embeddings")  # source of y/top2p/top5p

DATASETS = ["3gbn_h1", "3gbn_h9", "4fqi_h1", "4fqi_h3",
            "aayl49", "aayl49_ml", "aayl50", "aayl51", "aayl52"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Featuriser classes ──────────────────────────────────────────────────────

class ProtBertFeaturiser:
    name = "protbert"
    dim = 1024

    def __init__(self, batch_size=16):
        from transformers import BertModel, BertTokenizer
        self.tok = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.model = BertModel.from_pretrained("Rostlab/prot_bert").eval().to(DEVICE)
        self.batch_size = batch_size

    def transform(self, seqs: list[str]) -> np.ndarray:
        return self._embed_chain(seqs, desc="ProtBert")

    def transform_paired(self, vh_seqs: list[str], vl_seqs: list[str]) -> np.ndarray:
        """Separate-encode-then-concat: ProtBert has no native paired mode, so
        this mirrors the S1-Dual pattern (encode VH and VL independently,
        concatenate the two pooled embeddings) rather than naive string concat."""
        vh_emb = self._embed_chain(vh_seqs, desc="ProtBert-VH")
        vl_emb = self._embed_chain(vl_seqs, desc="ProtBert-VL")
        return np.concatenate([vh_emb, vl_emb], axis=1)

    def _embed_chain(self, seqs: list[str], desc: str) -> np.ndarray:
        out = []
        for start in tqdm(range(0, len(seqs), self.batch_size), desc=desc):
            batch = seqs[start:start + self.batch_size]
            spaced = [" ".join(s[:1000]) for s in batch]
            inputs = self.tok(spaced, return_tensors="pt", padding=True, truncation=True, max_length=1002)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                hidden = self.model(**inputs).last_hidden_state  # [B, L, 1024]
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            # exclude special tokens (CLS at 0, SEP at last valid pos) via mask on real residues only
            mask[:, 0, :] = 0  # CLS
            seq_lens = inputs["attention_mask"].sum(1)
            for i, L in enumerate(seq_lens):
                mask[i, L - 1, :] = 0  # SEP
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)


class AbLang2Featuriser:
    name = "ablang2"
    dim = 480

    def __init__(self, batch_size=32):
        import ablang2
        self.model = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False,
                                         ncpu=1, device=str(DEVICE))
        self.batch_size = batch_size

    def transform_paired(self, vh_seqs: list[str], vl_seqs: list[str]) -> np.ndarray:
        seqs = [f"{vh[:512]}|{vl[:512]}" for vh, vl in zip(vh_seqs, vl_seqs)]
        out = []
        for start in tqdm(range(0, len(seqs), self.batch_size), desc="AbLang2"):
            batch = seqs[start:start + self.batch_size]
            emb = self.model.seqcoding(batch)
            out.append(emb.astype(np.float32))
        return np.concatenate(out, axis=0)


class AntiBERTyFeaturiser:
    name = "antiberty"
    dim = 512

    def __init__(self, batch_size=32):
        from transformers import BertModel, BertTokenizer
        ckpt = Path(r"C:\Users\hs494\AppData\Local\anaconda3\envs\al\Lib\site-packages\antiberty\trained_models\AntiBERTy_md_smooth")
        vocab = Path(r"C:\Users\hs494\AppData\Local\anaconda3\envs\al\Lib\site-packages\antiberty\trained_models\vocab.txt")
        self.tok = BertTokenizer(str(vocab), do_lower_case=False)
        self.model = BertModel.from_pretrained(str(ckpt)).eval().to(DEVICE)
        self.dim = self.model.config.hidden_size
        self.batch_size = batch_size

    def transform(self, seqs: list[str]) -> np.ndarray:
        return self._embed_chain(seqs, desc="AntiBERTy")

    def transform_paired(self, vh_seqs: list[str], vl_seqs: list[str]) -> np.ndarray:
        """Separate-encode-then-concat (S1-Dual pattern); AntiBERTy is trained
        on single chains, so unlike AbLang2 it has no native paired-chain mode."""
        vh_emb = self._embed_chain(vh_seqs, desc="AntiBERTy-VH")
        vl_emb = self._embed_chain(vl_seqs, desc="AntiBERTy-VL")
        return np.concatenate([vh_emb, vl_emb], axis=1)

    def _embed_chain(self, seqs: list[str], desc: str) -> np.ndarray:
        out = []
        for start in tqdm(range(0, len(seqs), self.batch_size), desc=desc):
            batch = seqs[start:start + self.batch_size]
            spaced = [" ".join(s[:512]) for s in batch]
            inputs = self.tok(spaced, return_tensors="pt", padding=True, truncation=True, max_length=514)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                hidden = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            mask[:, 0, :] = 0
            seq_lens = inputs["attention_mask"].sum(1)
            for i, L in enumerate(seq_lens):
                mask[i, L - 1, :] = 0
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)


class ProGen2Featuriser:
    name = "progen2"
    dim = 1024

    def __init__(self, batch_size=64):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained("hugohrban/progen2-small")
        self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            "hugohrban/progen2-small", trust_remote_code=True).eval().to(DEVICE)
        # Compat shims for this transformers/torch version: (1) custom remote
        # code calls the removed PreTrainedModel.get_head_mask; (2) scale_attn
        # is a plain tensor attribute (not a registered buffer), so it's
        # never moved off the meta device used during fast init - both are
        # deterministic functions of config, safe to recompute directly.
        no_head_mask = lambda head_mask, num_layers: [None] * num_layers
        self.model.get_head_mask = no_head_mask
        self.model.transformer.get_head_mask = no_head_mask
        for blk in self.model.transformer.h:
            blk.attn.scale_attn = torch.sqrt(torch.tensor(blk.attn.head_dim, dtype=torch.float32)).to(DEVICE)
        self.dim = self.model.config.embed_dim if hasattr(self.model.config, "embed_dim") else 1024
        self.batch_size = batch_size

    def transform(self, seqs: list[str]) -> np.ndarray:
        return self._embed_chain(seqs, desc="ProGen2")

    def transform_paired(self, vh_seqs: list[str], vl_seqs: list[str]) -> np.ndarray:
        """Separate-encode-then-concat (S1-Dual pattern); ProGen2 is a causal
        LM over single chains with no native paired-chain mode."""
        vh_emb = self._embed_chain(vh_seqs, desc="ProGen2-VH")
        vl_emb = self._embed_chain(vl_seqs, desc="ProGen2-VL")
        return np.concatenate([vh_emb, vl_emb], axis=1)

    def _embed_chain(self, seqs: list[str], desc: str) -> np.ndarray:
        out = []
        for start in tqdm(range(0, len(seqs), self.batch_size), desc=desc):
            batch = [s[:1000] for s in seqs[start:start + self.batch_size]]
            inputs = self.tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1002)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                hidden = self.model(**inputs, output_hidden_states=True).hidden_states[-1]
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)


PLMS = {
    "antiberty": AntiBERTyFeaturiser,   # fastest (~0.015s/seq) - run first
    "ablang2": AbLang2Featuriser,       # ~0.042s/seq
    "progen2": ProGen2Featuriser,       # ~0.1s/seq
    "protbert": ProtBertFeaturiser,     # slowest (~0.19s/seq) - run last
}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--plms", nargs="+", default=list(PLMS.keys()), choices=list(PLMS.keys()))
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--paired", action="store_true",
                    help="Use separate-VH-then-VL-encode-and-concat instead of naive "
                         "VH+G+VL string concat. AbLang2 always uses its native paired "
                         "mode regardless of this flag (it has no naive-concat option).")
    args = p.parse_args()

    for plm_name in args.plms:
        suffix = "_paired" if (args.paired and plm_name != "ablang2") else ""
        out_root = Path(rf"D:\explainable_AL\data\embeddings_{plm_name}{suffix}")
        print(f"\n{'='*60}\nPLM: {plm_name}{suffix}\n{'='*60}")
        featuriser = PLMS[plm_name]()

        for ds in args.datasets:
            out_dir = out_root / ds
            if (out_dir / "X.npy").exists():
                print(f"  SKIP {ds} (already done)")
                continue

            p_path = PROCESSED / f"{ds}.parquet"
            esm2_dir = ESM2_EMB / ds
            if not p_path.exists() or not esm2_dir.exists():
                print(f"  SKIP {ds} (missing source data)")
                continue

            df = pd.read_parquet(p_path)
            print(f"  {ds}: {len(df):,} sequences")

            if plm_name == "ablang2" or args.paired:
                X = featuriser.transform_paired(df["vh_sequence"].tolist(), df["vl_sequence"].tolist())
            else:
                X = featuriser.transform(df["sequence"].tolist())

            y = np.load(esm2_dir / "y.npy")
            top2p = np.load(esm2_dir / "top2p.npy")
            top5p = np.load(esm2_dir / "top5p.npy")
            assert len(X) == len(y), f"{plm_name}/{ds}: X={len(X)} y={len(y)}"

            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "X.npy", X)
            np.save(out_dir / "y.npy", y)
            np.save(out_dir / "top2p.npy", top2p)
            np.save(out_dir / "top5p.npy", top5p)
            with open(out_dir / "meta.json", "w") as f:
                json.dump({"plm": plm_name, "dataset": ds, "n": len(df), "dim": int(X.shape[1])}, f, indent=2)
            print(f"    OK X={X.shape} -> {out_dir}/")


if __name__ == "__main__":
    main()
