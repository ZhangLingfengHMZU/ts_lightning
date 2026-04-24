#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from src.data_pipeline.dreamer.mat_io import DreamerMatIO, WindowSpec


def build_global_manifest(
    io: DreamerMatIO,
    window_spec: WindowSpec,
    is_force_rebuild: bool = False,
    spec_version: str = "v1",
) -> pd.DataFrame:

    out_dir = Path("data/dreamer/processed/manifests")
    out_dir.mkdir(parents=True, exist_ok=True)
    ws = str(window_spec.win_sec).replace(".", "p")
    ss = str(window_spec.stride_sec).replace(".", "p")
    out_path = out_dir / f"manifest__dreamer__ws{ws}__ss{ss}__v{spec_version}.parquet"
    if out_path.is_file() and not is_force_rebuild:
        print(f"[manifest] load cache: {out_path}")
        return pd.read_parquet(out_path)

    created_at_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    rows: List[Dict[str, Any]] = []

    n_subjects = io.num_subjects()
    print(f"[info] num_subjects={n_subjects}")

    for sid in range(n_subjects):
        n_trials = io.num_trials(sid)
        print(f"[info] sid={sid} n_trials={n_trials}")

        for trial_id in range(n_trials):
            labels = io.get_trial_labels(sid, trial_id)
            n_w = io.num_windows(sid, trial_id, window_spec)

            for window_id in range(n_w):
                # 这里调用 get_window 是为了复用你已有的时间对齐逻辑与meta
                sample = io.get_window(
                    sid=sid,
                    trial_id=trial_id,
                    window_id=window_id,
                    spec=window_spec,
                    channels_first=True,
                    dtype=np.float32,
                )
                meta = sample["meta"]

                rows.append(
                    {
                        # 稳定主键（逻辑主键）
                        "sample_id": f"sid{sid}_t{trial_id}_w{window_id}",
                        "sid": sid,
                        "trial_id": trial_id,
                        "window_id": window_id,

                        # 时间与索引信息
                        "start_sec": float(meta["start_sec"]),
                        "end_sec": float(meta["end_sec"]),
                        "eeg_start": int(meta["eeg_start"]),
                        "eeg_end": int(meta["eeg_end"]),
                        "ecg_start": int(meta["ecg_start"]),
                        "ecg_end": int(meta["ecg_end"]),

                        # 标签（先保留原始回归标签）
                        "valence": float(labels["valence"]),
                        "arousal": float(labels["arousal"]),
                        "dominance": float(labels["dominance"]),

                        # 配置与追踪字段
                        "win_sec": float(window_spec.win_sec),
                        "stride_sec": float(window_spec.stride_sec),
                        "eeg_fs": int(io.eeg_fs),
                        "ecg_fs": int(io.ecg_fs),
                        "spec_version": spec_version,
                        "created_at_utc": created_at_utc,
                    }
                )

    df = pd.DataFrame(rows)

    # 确保顺序稳定，再生成 row_id（物理索引主键）
    df = df.sort_values(["sid", "trial_id", "window_id"]).reset_index(drop=True)
    df.insert(0, "row_id", df.index.astype("int64"))

    save_manifest(df, out_path)

    return df


def save_manifest(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = out_path.suffix.lower()
    if suffix == ".parquet":
        # 需要 pyarrow 或 fastparquet
        df.to_parquet(out_path, index=False)
    elif suffix == ".csv":
        df.to_csv(out_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {suffix}. Use .parquet or .csv")

    print(f"[ok] saved: {out_path}")
    print(f"[ok] rows={len(df)}, cols={len(df.columns)}")