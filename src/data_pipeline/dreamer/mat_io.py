from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import math
import numpy as np
import scipy.io as sio


@dataclass(frozen=True)
class WindowSpec:
    win_sec: float
    stride_sec: float


class DreamerMatIO:
    """
    Thin adapter over DREAMER.mat so downstream code never touches
    nested [0,0]["..."] indexing directly.
    """

    def __init__(
        self,
        mat_path: str,
        eeg_fs: int = 128,
        ecg_fs: int = 256,
        verify_compressed_data_integrity: bool = False,
    ) -> None:
        self.mat_path = mat_path
        self.eeg_fs = int(eeg_fs)
        self.ecg_fs = int(ecg_fs)

        self._mat = sio.loadmat(
            mat_path,
            verify_compressed_data_integrity=verify_compressed_data_integrity,
        )

        # Core container: length == number of subjects (23 for DREAMER)
        # self._subjects = [subject0, subject1, ..., subject22]（语义上）
        self._subjects = self._mat["DREAMER"][0, 0]["Data"][0]

    # --------------------------
    # Basic metadata
    # --------------------------
    def num_subjects(self) -> int:
        return len(self._subjects)

    def num_trials(self, sid: int) -> int:
        s = self._get_subject(sid)
        return len(s["ScoreValence"][0, 0].flatten())

    # --------------------------
    # Public read APIs
    # --------------------------
    def get_trial_labels(self, sid: int, trial_id: int) -> Dict[str, float]:
        s = self._get_subject(sid)
        self._validate_trial_id(s, trial_id)

        valence = float(s["ScoreValence"][0, 0].flatten()[trial_id])
        arousal = float(s["ScoreArousal"][0, 0].flatten()[trial_id])
        dominance = float(s["ScoreDominance"][0, 0].flatten()[trial_id])

        return {
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
        }

    def get_trial_signals(self, sid: int, trial_id: int) -> Dict[str, np.ndarray]:
        s = self._get_subject(sid)
        self._validate_trial_id(s, trial_id)

        eeg_stim_trials = s["EEG"][0, 0]["stimuli"][0, 0]
        eeg_base_trials = s["EEG"][0, 0]["baseline"][0, 0]
        ecg_stim_trials = s["ECG"][0, 0]["stimuli"][0, 0]
        ecg_base_trials = s["ECG"][0, 0]["baseline"][0, 0]

        eeg_stim = eeg_stim_trials[trial_id, 0]  # (T_eeg, 14)
        eeg_base = eeg_base_trials[trial_id, 0]  # (B_eeg, 14)
        ecg_stim = ecg_stim_trials[trial_id, 0]  # (T_ecg, 2)
        ecg_base = ecg_base_trials[trial_id, 0]  # (B_ecg, 2)

        self._assert_duration_match(eeg_stim, ecg_stim, where=f"stim sid={sid}, trial={trial_id}")
        self._assert_duration_match(eeg_base, ecg_base, where=f"base sid={sid}, trial={trial_id}")

        return {
            "eeg_stim": eeg_stim,
            "ecg_stim": ecg_stim,
            "eeg_base": eeg_base,
            "ecg_base": ecg_base,
        }

    def num_windows(self, sid: int, trial_id: int, spec: WindowSpec) -> int:
        self._validate_spec(spec)

        signals = self.get_trial_signals(sid, trial_id)
        eeg = signals["eeg_stim"]  # (T_eeg, C)
        ecg = signals["ecg_stim"]  # (T_ecg, C)

        base_fs = math.lcm(self.eeg_fs, self.ecg_fs)

        # trial时长转为tick（取min更稳，防止极小不一致）
        eeg_ticks = (eeg.shape[0] * base_fs) // self.eeg_fs
        ecg_ticks = (ecg.shape[0] * base_fs) // self.ecg_fs
        trial_ticks = min(eeg_ticks, ecg_ticks)

        win_ticks = int(round(spec.win_sec * base_fs))
        step_ticks = int(round(spec.stride_sec * base_fs))

        if win_ticks <= 0:
            raise ValueError(f"win_sec too small or invalid: {spec.win_sec}")
        if step_ticks <= 0:
            raise ValueError(f"stride_sec too small or invalid: {spec.stride_sec}")

        if trial_ticks < win_ticks:
            return 0

        # 左闭右开窗口计数
        return 1 + (trial_ticks - win_ticks) // step_ticks

    def get_window(self, sid, trial_id, window_id, spec, channels_first=True, dtype=np.float32):
        self._validate_spec(spec)
        if window_id < 0:
            raise IndexError(f"window_id must be >= 0, got {window_id}")

        signals = self.get_trial_signals(sid, trial_id)
        labels = self.get_trial_labels(sid, trial_id)

        eeg = signals["eeg_stim"]  # (T_eeg, 14)
        ecg = signals["ecg_stim"]  # (T_ecg, 2)

        # ---- 关键：统一时间基准（整数tick），避免float round-trip误差 ----
        base_fs = math.lcm(self.eeg_fs, self.ecg_fs)   # 128/256 -> 256
        win_ticks = int(round(spec.win_sec * base_fs))
        step_ticks = int(round(spec.stride_sec * base_fs))

        if win_ticks <= 0 or step_ticks <= 0:
            raise ValueError(f"invalid window spec: win={spec.win_sec}, stride={spec.stride_sec}")

        start_tick = window_id * step_ticks
        end_tick = start_tick + win_ticks

        # 再映射回各模态索引（整数）
        start_eeg = start_tick * self.eeg_fs // base_fs
        end_eeg = end_tick * self.eeg_fs // base_fs
        start_ecg = start_tick * self.ecg_fs // base_fs
        end_ecg = end_tick * self.ecg_fs // base_fs

        if end_eeg > eeg.shape[0]:
            n_w = self.num_windows(sid, trial_id, spec)
            raise IndexError(f"EEG slice out of range: [{start_eeg}:{end_eeg}], eeg_len={eeg.shape[0]}")
        if end_ecg > ecg.shape[0]:
            raise IndexError(f"ECG slice out of range: [{start_ecg}:{end_ecg}], ecg_len={ecg.shape[0]}")

        # ---- 修复你现在的bug：明确计算 start_sec / end_sec ----
        start_sec = start_tick / base_fs
        end_sec = end_tick / base_fs

        # 对齐安全断言（建议保留）
        eeg_dur = (end_eeg - start_eeg) / self.eeg_fs
        ecg_dur = (end_ecg - start_ecg) / self.ecg_fs
        if abs(eeg_dur - ecg_dur) > 1e-9:
            raise AssertionError(f"duration mismatch: eeg={eeg_dur}, ecg={ecg_dur}")

        eeg_win = eeg[start_eeg:end_eeg, :]
        ecg_win = ecg[start_ecg:end_ecg, :]

        if channels_first:
            eeg_win = eeg_win.T
            ecg_win = ecg_win.T

        eeg_win = eeg_win.astype(dtype, copy=False)
        ecg_win = ecg_win.astype(dtype, copy=False)

        return {
            "eeg": eeg_win,
            "ecg": ecg_win,
            "label": labels,
            "meta": {
                "sid": sid,
                "trial_id": trial_id,
                "window_id": window_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "eeg_start": start_eeg,
                "eeg_end": end_eeg,
                "ecg_start": start_ecg,
                "ecg_end": end_ecg,
                "eeg_fs": self.eeg_fs,
                "ecg_fs": self.ecg_fs,
                "win_sec": spec.win_sec,
                "stride_sec": spec.stride_sec,
            },
        }

    # --------------------------
    # Internal helpers
    # --------------------------
    def _get_subject(self, sid: int):
        if sid < 0 or sid >= self.num_subjects():
            raise IndexError(f"sid out of range: {sid}, num_subjects={self.num_subjects()}")
        return self._subjects[sid]

    @staticmethod
    def _validate_trial_id(subject_struct, trial_id: int) -> None:
        n_trials = len(subject_struct["ScoreValence"][0, 0].flatten())
        if trial_id < 0 or trial_id >= n_trials:
            raise IndexError(f"trial_id out of range: {trial_id}, n_trials={n_trials}")

    @staticmethod
    def _validate_spec(spec: WindowSpec) -> None:
        if spec.win_sec <= 0:
            raise ValueError(f"win_sec must be > 0, got {spec.win_sec}")
        if spec.stride_sec <= 0:
            raise ValueError(f"stride_sec must be > 0, got {spec.stride_sec}")

    def _assert_duration_match(self, eeg_arr: np.ndarray, ecg_arr: np.ndarray, where: str = "") -> None:
        eeg_sec = eeg_arr.shape[0] / self.eeg_fs
        ecg_sec = ecg_arr.shape[0] / self.ecg_fs
        if abs(eeg_sec - ecg_sec) > 1e-6:
            raise AssertionError(
                f"EEG/ECG duration mismatch at {where}: eeg_sec={eeg_sec}, ecg_sec={ecg_sec}"
            )