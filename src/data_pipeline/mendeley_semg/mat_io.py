from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import scipy.io as sio


@dataclass(frozen=True)
class SubjectSignal:
    """Container for one subject's multi-channel sEMG signal."""
    subject_id: int
    data: np.ndarray  # shape: [time, channels]
    fs: int
    units: list[str]
    source: str
    file_path: Path


class MendeleySemgMatIO:
    """
    Minimal, explicit IO adapter for Mendeley sEMG MAT files.

    Scope of this MVP:
    - Load exactly one subject from raw/filtered MAT.
    - Return clean and explicit metadata.
    - No segmentation/windowing yet.
    """

    def __init__(self, dataset_root: str, source: str = "filtered") -> None:
        """
        Args:
            dataset_root: e.g. "data/Mendeley_semg/raw/sEMG-dataset"
            source: "raw" or "filtered"
        """
        self.dataset_root = Path(dataset_root)
        self.source = source.strip().lower()
        self._REP_START_SEC = (4.0, 138.0, 272.0, 406.0, 540.0)
        self._GESTURE_NAMES = (
            "REST", "EXTENSION", "FLEXION", "ULNAR_DEVIATION", "RADIAL_DEVIATION",
            "GRIP", "ABDUCTION", "ADDUCTION", "SUPINATION", "PRONATION",
        )

        if self.source not in {"raw", "filtered"}:
            raise ValueError(f"source must be 'raw' or 'filtered', got: {source}")

        self._mat_dir = self.dataset_root / self.source / "mat"
        if not self._mat_dir.is_dir():
            raise FileNotFoundError(f"MAT directory not found: {self._mat_dir}")

    def get_subject_signal(self, subject_id: int) -> SubjectSignal:
        """
        Load one subject's full recording.

        Args:
            subject_id: integer in [1, 40]

        Returns:
            SubjectSignal with:
              - data shape [T, 4]
              - fs=2000 (from file)
              - units (usually mV)
        """
        self._validate_subject_id(subject_id)

        mat_path = self._build_mat_path(subject_id)
        raw_obj = sio.loadmat(mat_path, verify_compressed_data_integrity=False)
        parsed = self._parse_mat_object(raw_obj, mat_path)

        return SubjectSignal(
            subject_id=subject_id,
            data=parsed["data"],
            fs=parsed["fs"],
            units=parsed["units"],
            source=self.source,
            file_path=mat_path,
        )

    def get_segment(
        self,
        subject_id: int,
        rep_idx: int,
        gesture_idx: int,
        seg_start_sec: float = 2.0,
        seg_end_sec: float = 8.0,
    ) -> Dict[str, Any]:
        """
        Return one gesture segment from one subject.

        Protocol is aligned with official Mendeley segmentation script:
        rep start sec = [4, 138, 272, 406, 540], each gesture block = 10 sec.
        """
        # 校验输入参数
        self._validate_subject_id(subject_id)
        self._validate_rep_idx(rep_idx)
        self._validate_gesture_idx(gesture_idx)
        self._validate_segment_seconds(seg_start_sec, seg_end_sec)

        # 读取某subject完整信号
        subject = self.get_subject_signal(subject_id)
        fs = subject.fs

        # 找到具体sub, rep, gesture的开始和结束时间
        rep_start_sec = self._REP_START_SEC[rep_idx]
        gesture_block_start_sec = rep_start_sec + 10.0 * gesture_idx
        abs_start_sec = gesture_block_start_sec + seg_start_sec
        abs_end_sec = gesture_block_start_sec + seg_end_sec
        start_sample = int(round(abs_start_sec * fs))
        end_sample = int(round(abs_end_sec * fs))

        if start_sample < 0:
            raise ValueError(f"start_sample < 0: {start_sample}")
        if end_sample > subject.data.shape[0]:
            raise ValueError(
                f"end_sample out of range: {end_sample}, signal_len={subject.data.shape[0]}"
            )
        if end_sample <= start_sample:
            raise ValueError(
                f"Invalid segment range: start={start_sample}, end={end_sample}"
            )

        x = subject.data[start_sample:end_sample, :]
        if x.ndim != 2 or x.shape[1] != 4:
            raise ValueError(f"Segment shape must be [T,4], got {x.shape}")

        return {
            "x": x,  # shape [T, 4], float32
            "label": gesture_idx,
            "gesture_name": self._GESTURE_NAMES[gesture_idx],
            "meta": {
                "subject_id": subject_id,
                "rep_idx": rep_idx,
                "gesture_idx": gesture_idx,
                "source": self.source,
                "fs": fs,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "start_sec": abs_start_sec,
                "end_sec": abs_end_sec,
                "units": subject.units,
                "file_path": str(subject.file_path),
            },
        }

    @staticmethod
    def _validate_rep_idx(rep_idx: int) -> None:
        if not isinstance(rep_idx, int):
            raise TypeError(f"rep_idx must be int, got: {type(rep_idx)}")
        if rep_idx < 0 or rep_idx > 4:
            raise ValueError(f"rep_idx out of range [0,4]: {rep_idx}")

    @staticmethod
    def _validate_gesture_idx(gesture_idx: int) -> None:
        if not isinstance(gesture_idx, int):
            raise TypeError(f"gesture_idx must be int, got: {type(gesture_idx)}")
        if gesture_idx < 0 or gesture_idx > 9:
            raise ValueError(f"gesture_idx out of range [0,9]: {gesture_idx}")

    @staticmethod
    def _validate_segment_seconds(seg_start_sec: float, seg_end_sec: float) -> None:
        if seg_start_sec < 0:
            raise ValueError(f"seg_start_sec must be >= 0, got: {seg_start_sec}")
        if seg_end_sec <= seg_start_sec:
            raise ValueError(
                f"seg_end_sec must be > seg_start_sec, got: {seg_start_sec}, {seg_end_sec}"
            )
        if seg_end_sec > 10.0:
            raise ValueError(f"seg_end_sec must be <= 10 (gesture block length), got: {seg_end_sec}")

    def _build_mat_path(self, subject_id: int) -> Path:
        filename = f"{subject_id}_{self.source}.mat"
        mat_path = self._mat_dir / filename
        if not mat_path.exists():
            raise FileNotFoundError(f"Subject MAT file not found: {mat_path}")
        return mat_path

    @staticmethod
    def _validate_subject_id(subject_id: int) -> None:
        if not isinstance(subject_id, int):
            raise TypeError(f"subject_id must be int, got: {type(subject_id)}")
        if subject_id < 1 or subject_id > 40:
            raise ValueError(f"subject_id out of range [1,40]: {subject_id}")

    @staticmethod
    def _parse_mat_object(raw_obj: Dict[str, Any], mat_path: Path) -> Dict[str, Any]:
        required_keys = {"data", "fs", "units"}
        missing = required_keys - set(raw_obj.keys())
        if missing:
            raise KeyError(f"Missing keys {missing} in MAT file: {mat_path}")

        data = np.asarray(raw_obj["data"])
        if data.ndim != 2:
            raise ValueError(f"Expected 2D data array, got shape {data.shape} in {mat_path}")
        if data.shape[1] != 4:
            raise ValueError(f"Expected 4 channels, got {data.shape[1]} in {mat_path}")
        # Make downstream behavior predictable
        data = data.astype(np.float32, copy=False)

        fs_arr = np.asarray(raw_obj["fs"]).reshape(-1)
        if fs_arr.size == 0:
            raise ValueError(f"Invalid fs in MAT file: {mat_path}")
        fs = int(fs_arr[0])

        units_raw = raw_obj["units"]
        units_arr = np.asarray(units_raw).reshape(-1)
        units = [str(u).strip() for u in units_arr.tolist()]

        if len(units) != 4:
            raise ValueError(f"Expected 4 channel units, got {len(units)} in {mat_path}")
        if any(not u for u in units):
            raise ValueError(f"Empty unit string found in {mat_path}")

        return {"data": data, "fs": fs, "units": units}
    @staticmethod
    def _validate_rep_idx(rep_idx: int) -> None:
        if not isinstance(rep_idx, int):
            raise TypeError(f"rep_idx must be int, got: {type(rep_idx)}")
        if rep_idx < 0 or rep_idx > 4:
            raise ValueError(f"rep_idx out of range [0,4]: {rep_idx}")
    @staticmethod
    def _validate_gesture_idx(gesture_idx: int) -> None:
        if not isinstance(gesture_idx, int):
            raise TypeError(f"gesture_idx must be int, got: {type(gesture_idx)}")
        if gesture_idx < 0 or gesture_idx > 9:
            raise ValueError(f"gesture_idx out of range [0,9]: {gesture_idx}")
    @staticmethod
    def _validate_segment_seconds(seg_start_sec: float, seg_end_sec: float) -> None:
        if seg_start_sec < 0:
            raise ValueError(f"seg_start_sec must be >= 0, got: {seg_start_sec}")
        if seg_end_sec <= seg_start_sec:
            raise ValueError(
                f"seg_end_sec must be > seg_start_sec, got: {seg_start_sec}, {seg_end_sec}"
            )
        if seg_end_sec > 10.0:
            raise ValueError(f"seg_end_sec must be <= 10 (gesture block length), got: {seg_end_sec}")