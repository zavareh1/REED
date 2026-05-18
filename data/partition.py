import numpy as np
from typing import List

def _to_list_arrays(parts: List[np.ndarray]) -> List[np.ndarray]:
    return [np.asarray(p, dtype=int) for p in parts]

def iid_splits(num_samples: int, K: int, rng) -> list[np.ndarray]:
    idx = np.arange(num_samples)
    rng.shuffle(idx)
    parts = np.array_split(idx, K)
    return _to_list_arrays(parts)

def dirichlet_splits(labels: np.ndarray, K: int, alpha: float, rng, min_size: int = 1) -> list[np.ndarray]:
    """Label-aware Dirichlet split (non-IID). Ensures no empty client by retrying small draws."""
    classes = np.unique(labels)
    idxs = [np.where(labels == c)[0] for c in classes]
    parts = [list() for _ in range(K)]
    for c_idxs in idxs:
        props = rng.dirichlet([alpha] * K)
        props = np.maximum(props, 1e-12); props = props / props.sum()
        alloc = (props * len(c_idxs)).astype(int)
        while alloc.sum() < len(c_idxs):
            alloc[rng.integers(0, K)] += 1
        rng.shuffle(c_idxs)
        offs = np.cumsum(np.concatenate([[0], alloc]))
        for k in range(K):
            parts[k].extend(c_idxs[offs[k]:offs[k+1]])
    parts = [np.array(p, dtype=int) for p in parts]
    # Rebalance tiny partitions to reach min_size
    for k in range(K):
        if parts[k].size < min_size:
            need = min_size - parts[k].size
            donors = [j for j in range(K) if parts[j].size > min_size]
            for _ in range(need):
                if not donors: break
                j = int(np.random.choice(donors))
                take_idx = int(np.random.randint(0, parts[j].size))
                parts[k] = np.append(parts[k], parts[j][take_idx])
                parts[j] = np.delete(parts[j], take_idx)
                if parts[j].size <= min_size and j in donors:
                    donors.remove(j)
    return parts

def shards_partition(labels: np.ndarray, K: int, shards_per_client: int, rng) -> list[np.ndarray]:
    """Classic 'pathological' non-IID: sort by label, split into equal shards, assign shards to clients."""
    num_samples = len(labels)
    num_shards = K * shards_per_client
    shard_size = max(1, num_samples // num_shards)
    # sort indices by label
    order = np.argsort(labels, kind="stable")
    trimmed = order[:shard_size * num_shards]  # drop tail if needed
    shards = [trimmed[i*shard_size:(i+1)*shard_size] for i in range(num_shards)]
    rng.shuffle(shards)
    parts = [np.concatenate(shards[i*shards_per_client:(i+1)*shards_per_client]) for i in range(K)]
    # distribute leftover (if any)
    leftover = order[shard_size * num_shards:]
    if leftover.size > 0:
        for i, idx in enumerate(leftover):
            parts[i % K] = np.append(parts[i % K], idx)
    return _to_list_arrays(parts)

def label_skew_n_classes(labels: np.ndarray, K: int, n_classes_per_client: int, rng) -> list[np.ndarray]:
    """Each client is restricted to n_classes_per_client labels; all data from chosen labels divided among chosen clients."""
    classes = np.unique(labels)
    C = len(classes)
    if n_classes_per_client > C:
        raise ValueError("n_classes_per_client cannot exceed number of classes")
    chosen = [rng.choice(classes, size=n_classes_per_client, replace=False) for _ in range(K)]
    holders = {c: [k for k in range(K) if c in chosen[k]] for c in classes}
    parts = [list() for _ in range(K)]
    for c in classes:
        idxs = np.where(labels == c)[0]
        rng.shuffle(idxs)
        H = holders[c] or list(range(K))
        splits = np.array_split(idxs, len(H))
        for h, chunk in zip(H, splits):
            parts[h].extend(chunk.tolist())
    return [np.array(p, dtype=int) for p in parts]

def quantity_skew(num_samples: int, K: int, rng, alpha: float = 0.3, min_size: int = 10) -> list[np.ndarray]:
    """Client sizes follow a Dirichlet over counts, ignoring labels (quantity skew)."""
    idx = np.arange(num_samples); rng.shuffle(idx)
    props = rng.dirichlet([alpha] * K)
    counts = (props * num_samples).astype(int)
    while counts.sum() < num_samples: counts[rng.integers(0, K)] += 1
    for k in range(K):
        if counts[k] < min_size:
            deficit = min_size - counts[k]
            donors = [j for j in range(K) if counts[j] > min_size]
            for _ in range(deficit):
                if not donors: break
                j = int(np.random.choice(donors))
                counts[j] -= 1; counts[k] += 1
                if counts[j] <= min_size: donors.remove(j)
    parts, off = [], 0
    for k in range(K):
        parts.append(idx[off:off+counts[k]]); off += counts[k]
    return _to_list_arrays(parts)

def majority_class_skew(labels: np.ndarray, K: int, rng, p_min: float = 0.8, p_max: float = 0.95) -> list[np.ndarray]:
    """Each client picks a majority class; majority share in [p_min,p_max] of its local data."""
    num_samples = len(labels)
    idx_all = np.arange(num_samples)
    classes = np.unique(labels)
    base = num_samples // K
    parts = []
    for _ in range(K):
        maj = int(np.random.choice(classes))
        p = float(np.random.uniform(p_min, p_max))
        maj_idx = idx_all[labels == maj].copy()
        non_idx = idx_all[labels != maj].copy()
        np.random.shuffle(maj_idx); np.random.shuffle(non_idx)
        n_maj = min(len(maj_idx), int(round(p * base)))
        sel = np.concatenate([maj_idx[:n_maj], non_idx[:base - n_maj]])
        parts.append(sel)
    used = np.concatenate(parts) if parts else np.array([], dtype=int)
    leftover = np.setdiff1d(idx_all, used, assume_unique=False)
    for i, idx in enumerate(leftover):
        parts[i % K] = np.append(parts[i % K], idx)
    return _to_list_arrays(parts)
