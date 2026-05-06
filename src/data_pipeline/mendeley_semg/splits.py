from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Set, Tuple
import random

import pandas as pd


@dataclass(frozen=True)
class SplitIndices:
    """
    Stable sample ids for each split.

    We intentionally keep this minimal so downstream Dataset code only
    depends on train/val/test indices and not on protocol details.
    """

    train_indices: Tuple[int, ...]
    val_indices: Tuple[int, ...]
    test_indices: Tuple[int, ...]


def build_subject_independent_indices(
    manifest_df: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    subject_col: str = "subject_id",
    index_col: str = "row_id",
) -> SplitIndices:
    """
    Subject-independent split:
    a subject appears in exactly one of train/val/test.

    Returns manifest row_id indices.
    """
    _require_columns(manifest_df, (subject_col, index_col))
    _validate_ratios(train_ratio, val_ratio)

    subject_ids = sorted({int(x) for x in manifest_df[subject_col].tolist()})
    if len(subject_ids) < 3:
        raise ValueError("Need at least 3 subjects for train/val/test split")

    rng = random.Random(seed)
    shuffled = subject_ids[:]
    rng.shuffle(shuffled)

    n_subjects = len(shuffled)
    n_train = int(round(n_subjects * train_ratio))
    n_val = int(round(n_subjects * val_ratio))

    # Keep all three splits non-empty.
    n_train = max(1, min(n_train, n_subjects - 2))
    n_val = max(1, min(n_val, n_subjects - n_train - 1))

    train_subjects = set(shuffled[:n_train])
    val_subjects = set(shuffled[n_train : n_train + n_val])
    test_subjects = set(shuffled[n_train + n_val :])
    _check_disjoint_and_cover(set(subject_ids), train_subjects, val_subjects, test_subjects)

    train_indices = _indices_by_membership(manifest_df, subject_col, train_subjects, index_col)
    val_indices = _indices_by_membership(manifest_df, subject_col, val_subjects, index_col)
    test_indices = _indices_by_membership(manifest_df, subject_col, test_subjects, index_col)

    _validate_index_partition(manifest_df, index_col, train_indices, val_indices, test_indices)
    return SplitIndices(train_indices=train_indices, val_indices=val_indices, test_indices=test_indices)


def build_subject_dependent_by_rep_indices(
    manifest_df: pd.DataFrame,
    train_reps: Sequence[int] = (0, 1, 2),
    val_reps: Sequence[int] = (3,),
    test_reps: Sequence[int] = (4,),
    rep_col: str = "rep_idx",
    index_col: str = "row_id",
) -> SplitIndices:
    """
    Subject-dependent split:
    each subject appears in train/val/test, split is done by repetition index.

    Returns manifest row_id indices.
    """
    _require_columns(manifest_df, (rep_col, index_col))

    all_reps = {int(x) for x in manifest_df[rep_col].tolist()}
    train_set = {int(x) for x in train_reps}
    val_set = {int(x) for x in val_reps}
    test_set = {int(x) for x in test_reps}

    _check_disjoint_and_cover(all_reps, train_set, val_set, test_set)

    train_indices = _indices_by_membership(manifest_df, rep_col, train_set, index_col)
    val_indices = _indices_by_membership(manifest_df, rep_col, val_set, index_col)
    test_indices = _indices_by_membership(manifest_df, rep_col, test_set, index_col)

    _validate_index_partition(manifest_df, index_col, train_indices, val_indices, test_indices)
    return SplitIndices(train_indices=train_indices, val_indices=val_indices, test_indices=test_indices)


def apply_split_column(
    manifest_df: pd.DataFrame,
    split: SplitIndices,
    index_col: str = "row_id",
    split_col: str = "split",
) -> pd.DataFrame:
    """
    Return a new DataFrame with a split column ("train"/"val"/"test").
    """
    _require_columns(manifest_df, (index_col,))

    out = manifest_df.copy()
    mapping = {idx: "train" for idx in split.train_indices}
    mapping.update({idx: "val" for idx in split.val_indices})
    mapping.update({idx: "test" for idx in split.test_indices})

    out[split_col] = out[index_col].map(mapping)
    if out[split_col].isna().any():
        missing = out.loc[out[split_col].isna(), index_col].tolist()[:10]
        raise ValueError(f"Split mapping missing indices, first few: {missing}")

    return out


def _indices_by_membership(
    manifest_df: pd.DataFrame,
    col_name: str,
    members: Set[int],
    index_col: str,
) -> Tuple[int, ...]:
    values = manifest_df[col_name].astype(int)
    rows = manifest_df.loc[values.isin(members), index_col].astype(int).tolist()
    return tuple(sorted(rows))


def _validate_ratios(train_ratio: float, val_ratio: float) -> None:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in [0,1), got {val_ratio}")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")


def _require_columns(manifest_df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in manifest_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _check_disjoint_and_cover(
    all_ids: Set[int],
    train_ids: Set[int],
    val_ids: Set[int],
    test_ids: Set[int],
) -> None:
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("train/val/test overlap detected")
    if (train_ids | val_ids | test_ids) != all_ids:
        raise ValueError("train/val/test do not cover all ids")


def _validate_index_partition(
    manifest_df: pd.DataFrame,
    index_col: str,
    train_indices: Tuple[int, ...],
    val_indices: Tuple[int, ...],
    test_indices: Tuple[int, ...],
) -> None:
    all_indices = {int(x) for x in manifest_df[index_col].tolist()}
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)
    _check_disjoint_and_cover(all_indices, train_set, val_set, test_set)