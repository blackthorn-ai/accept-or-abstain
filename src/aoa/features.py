"""Fingerprints, descriptors and Tanimoto nearest-neighbour signals."""
from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

DESCRIPTORS = [
    "MolWt",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "HeavyAtomCount",
    "NumHeteroatoms",
]
_DESC_FN = {name: fn for name, fn in Descriptors.descList if name in DESCRIPTORS}


def _gen(n_bits: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)


def ecfp_bits(smiles: list[str], n_bits: int = 2048) -> np.ndarray:
    """ECFP4 bit vectors as a uint8 matrix; an empty SMILES gives a zero row."""
    gen = _gen(n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    cache: dict[str, np.ndarray] = {}
    for i, s in enumerate(smiles):
        if not s:
            continue
        if s not in cache:
            m = Chem.MolFromSmiles(s)
            arr = np.zeros((n_bits,), dtype=np.uint8)
            if m is not None:
                DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m), arr)
            cache[s] = arr
        out[i] = cache[s]
    return out


def ecfp_counts(smiles: list[str], n_bits: int = 2048) -> np.ndarray:
    gen = _gen(n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fp = gen.GetCountFingerprintAsNumPy(m)
        out[i] = fp
    return out


def descriptors(smiles: list[str]) -> np.ndarray:
    out = np.zeros((len(smiles), len(DESCRIPTORS)), dtype=np.float32)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        out[i] = [_DESC_FN[d](m) for d in DESCRIPTORS]
    return out


def molecule_features(smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Model input (ECFP4 counts + descriptors) and similarity bits (ECFP4 bits)."""
    X = np.hstack([ecfp_counts(smiles), descriptors(smiles)])
    B = ecfp_bits(smiles)
    return X, B


def reaction_features(df, components: list[str], n_bits: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Concatenated per-component ECFP4 bits; used both as input and for similarity."""
    B = np.hstack([ecfp_bits(df[c].tolist(), n_bits) for c in components])
    return B.astype(np.float32), B


def tanimoto_topk(Q: np.ndarray, R: np.ndarray, k: int = 5, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Max and mean-of-top-k Tanimoto similarity of each query row to the reference rows.

    Q and R are 0/1 matrices over the same bits. Returns (s1, s_k).
    """
    Rf = R.astype(np.float32)
    r_cnt = Rf.sum(1)
    s1 = np.zeros(len(Q), dtype=np.float32)
    sk = np.zeros(len(Q), dtype=np.float32)
    for a in range(0, len(Q), chunk):
        q = Q[a : a + chunk].astype(np.float32)
        inter = q @ Rf.T
        union = q.sum(1)[:, None] + r_cnt[None, :] - inter
        sim = np.where(union > 0, inter / np.maximum(union, 1e-9), 1.0)
        kk = min(k, sim.shape[1])
        top = np.partition(sim, -kk, axis=1)[:, -kk:]
        s1[a : a + chunk] = top.max(1)
        sk[a : a + chunk] = top.mean(1)
    return s1, sk


def butina_clusters(B: np.ndarray, distance: float = 0.65, chunk: int = 2048) -> np.ndarray:
    """Butina clustering of 0/1 fingerprints at a Tanimoto distance cutoff.

    Implemented with neighbour lists so that memory stays linear in the number of
    neighbour pairs rather than quadratic in the number of molecules.
    """
    n = len(B)
    thr = 1.0 - distance
    Bf = B.astype(np.float32)
    cnt = Bf.sum(1)
    neigh: list[np.ndarray] = []
    for a in range(0, n, chunk):
        q = Bf[a : a + chunk]
        inter = q @ Bf.T
        union = cnt[a : a + chunk, None] + cnt[None, :] - inter
        sim = np.where(union > 0, inter / np.maximum(union, 1e-9), 1.0)
        for i in range(sim.shape[0]):
            nb = np.flatnonzero(sim[i] >= thr)
            neigh.append(nb[nb != a + i])
    order = np.argsort([-len(x) for x in neigh], kind="stable")
    labels = np.full(n, -1, dtype=np.int64)
    cid = 0
    for i in order:
        if labels[i] >= 0:
            continue
        members = neigh[i][labels[neigh[i]] < 0]
        labels[i] = cid
        labels[members] = cid
        cid += 1
    return labels
