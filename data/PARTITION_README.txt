Example usage (inside your script):

    from data.partition import (
        iid_splits, dirichlet_splits, shards_partition,
        label_skew_n_classes, quantity_skew, majority_class_skew
    )
    parts = shards_partition(labels=y, K=10, shards_per_client=2, rng=np.random.default_rng(123))

See main_mnist.py in the project root for a wired CLI version.
