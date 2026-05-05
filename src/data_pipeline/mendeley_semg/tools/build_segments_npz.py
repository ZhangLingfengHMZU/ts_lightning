from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.data_pipeline.mendeley_semg.mat_io import MendeleySemgMatIO


def _resolve_subject_ids(subject_ids_cfg: Any) -> List[int]:
    # 支持: "all" 或 [1,2,3]
    if isinstance(subject_ids_cfg, str) and subject_ids_cfg.lower() == "all":
        return list(range(1, 41))
    if isinstance(subject_ids_cfg, list):
        return [int(x) for x in subject_ids_cfg]
    raise ValueError(f"Invalid run.subject_ids: {subject_ids_cfg}")


@hydra.main(
    version_base=None,
    config_path="../../../../configs/mendeley_semg",
    config_name="build_segments_npz",
)
def main(cfg: DictConfig) -> None:
    print("[config]")
    print(OmegaConf.to_yaml(cfg))

    io = MendeleySemgMatIO(
        dataset_root=cfg.data.dataset_root,
        source=cfg.data.source,
    )

    subject_ids = _resolve_subject_ids(cfg.run.subject_ids)
    rep_indices = list(cfg.run.rep_indices)
    gesture_indices = list(cfg.run.gesture_indices)

    seg_root = Path(cfg.output.segment_root) / cfg.data.source
    seg_root.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(cfg.output.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    errors = []

    created_at_utc = datetime.now(timezone.utc).isoformat()
    seg_start_sec = float(cfg.segment.start_sec)
    seg_end_sec = float(cfg.segment.end_sec)

    for subject_id in subject_ids:
        for rep_idx in rep_indices:
            for gesture_idx in gesture_indices:
                sample_id = f"s{subject_id:02d}_r{rep_idx}_g{gesture_idx}"
                out_path = seg_root / f"{sample_id}.npz"

                try:
                    seg = io.get_segment(
                        subject_id=subject_id,
                        rep_idx=rep_idx,
                        gesture_idx=gesture_idx,
                        seg_start_sec=seg_start_sec,
                        seg_end_sec=seg_end_sec,
                    )
                    x = seg["x"]
                    meta = seg["meta"]

                    if out_path.exists() and not bool(cfg.run.overwrite):
                        pass
                    else:
                        np.savez_compressed(
                            out_path,
                            x=x,  # [T, 4], float32
                            label=np.int64(seg["label"]),
                            gesture_name=np.array(seg["gesture_name"]),
                            subject_id=np.int64(subject_id),
                            rep_idx=np.int64(rep_idx),
                            gesture_idx=np.int64(gesture_idx),
                            fs=np.int64(meta["fs"]),
                            start_sample=np.int64(meta["start_sample"]),
                            end_sample=np.int64(meta["end_sample"]),
                            start_sec=np.float32(meta["start_sec"]),
                            end_sec=np.float32(meta["end_sec"]),
                            source=np.array(meta["source"]),
                        )

                    rows.append(
                        {
                            "sample_id": sample_id,
                            "source": cfg.data.source,
                            "subject_id": subject_id,
                            "rep_idx": rep_idx,
                            "gesture_idx": gesture_idx,
                            "gesture_name": seg["gesture_name"],
                            "seg_start_sec": seg_start_sec,
                            "seg_end_sec": seg_end_sec,
                            "fs": int(meta["fs"]),
                            "start_sample": int(meta["start_sample"]),
                            "end_sample": int(meta["end_sample"]),
                            "num_samples": int(x.shape[0]),
                            "num_channels": int(x.shape[1]),
                            "path": str(out_path),
                            "created_at_utc": created_at_utc,
                        }
                    )

                except Exception as e:
                    errors.append(
                        {
                            "sample_id": sample_id,
                            "subject_id": subject_id,
                            "rep_idx": rep_idx,
                            "gesture_idx": gesture_idx,
                            "error": repr(e),
                        }
                    )

    df = pd.DataFrame(rows).sort_values(
        ["subject_id", "rep_idx", "gesture_idx"]
    ).reset_index(drop=True)
    df.insert(0, "row_id", df.index.astype("int64"))

    if manifest_path.suffix.lower() == ".csv":
        df.to_csv(manifest_path, index=False)
    elif manifest_path.suffix.lower() == ".parquet":
        df.to_parquet(manifest_path, index=False)
    else:
        raise ValueError("manifest_path must end with .csv or .parquet")

    print(f"[ok] saved manifest: {manifest_path}")
    print(f"[ok] rows={len(df)}")

    if errors:
        err_path = manifest_path.with_name(manifest_path.stem + "__errors.csv")
        pd.DataFrame(errors).to_csv(err_path, index=False)
        print(f"[warn] errors={len(errors)}, saved to: {err_path}")
    else:
        print("[ok] errors=0")


if __name__ == "__main__":
    main()