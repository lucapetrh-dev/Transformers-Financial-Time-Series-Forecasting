from __future__ import annotations

from collections.abc import Iterator


def walk_forward_splits(
    n_samples: int,
    min_train_size: int,
    test_size: int,
    step_size: int | None = None,
    expanding: bool = True,
) -> Iterator[tuple[list[int], list[int]]]:
    if n_samples <= 0:
        return
    if test_size <= 0 or min_train_size <= 0:
        raise ValueError("test_size and min_train_size must be positive")

    if step_size is None:
        step_size = test_size

    train_start = 0
    train_end = min_train_size

    while train_end + test_size <= n_samples:
        test_start = train_end
        test_end = test_start + test_size

        if expanding:
            train_idx = list(range(0, train_end))
        else:
            train_idx = list(range(train_start, train_end))

        test_idx = list(range(test_start, test_end))
        yield train_idx, test_idx

        if expanding:
            train_end += step_size
        else:
            train_start += step_size
            train_end += step_size


def purged_kfold_splits(
    n_samples: int,
    n_splits: int = 5,
    embargo: int = 0,
    label_horizon: int = 1,
):
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    fold_size = n_samples // n_splits
    if fold_size == 0:
        raise ValueError("n_splits too large for available samples")

    for fold in range(n_splits):
        val_start = fold * fold_size
        val_end = n_samples if fold == n_splits - 1 else (fold + 1) * fold_size

        purge_start = max(0, val_start - label_horizon)
        purge_end = min(n_samples, val_end + embargo)

        val_idx = list(range(val_start, val_end))
        train_idx = [i for i in range(n_samples) if i < purge_start or i >= purge_end]

        yield train_idx, val_idx
