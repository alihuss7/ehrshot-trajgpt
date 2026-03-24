from __future__ import annotations

"""TrajGPT patient embedder for EHRSHOT evaluation."""

import bisect
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from models.embedder import PatientEmbedder
from models.trajgpt.model import TrajGPT
from models.trajgpt.tokenizer import EHRTokenizer


class TrajGPTEmbedder(PatientEmbedder):
    """TrajGPT embedder (time-specific last-state representation)."""

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
        device: str = "cpu",
        batch_size: int = 64,
        max_seq_len: int = 256,
    ):
        self.device = self._resolve_device(device)
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        self.tokenizer = EHRTokenizer.load(tokenizer_path)

        print(f"Loading TrajGPT from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = checkpoint.get("config", {})

        self._d_model = config.get("d_model", 200)
        self.model = TrajGPT(
            vocab_size=self.tokenizer.vocab_size,
            d_model=self._d_model,
            qk_dim=config.get("qk_dim", config.get("d_model", 200)),
            v_dim=config.get("v_dim", 400),
            ff_dim=config.get("ff_dim", 800),
            num_layers=config.get("num_layers", 8),
            num_heads=config.get("num_heads", 4),
            tau=config.get("tau", 20.0),
            dropout=0.0,
            max_seq_len=max_seq_len,
            pad_id=self.tokenizer.pad_id,
            sos_id=self.tokenizer.sos_id,
            forecast_method=config.get("forecast_method", "time_specific"),
            use_bias_in_sra=config.get("use_bias_in_sra", False),
            use_bias_in_mlp=config.get("use_bias_in_mlp", True),
            use_bias_in_sra_out=config.get("use_bias_in_sra_out", False),
            use_default_gamma=config.get("use_default_gamma", False),
            output_retentions=config.get("output_retentions", False),
            use_cache=config.get("use_cache", True),
            forward_impl=config.get("forward_impl", "parallel"),
        ).to(self.device)

        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as e:
            raise RuntimeError(
                "Checkpoint architecture mismatch with strict TrajGPT mode. "
                "This checkpoint was trained with an older model variant. "
                "Rerun `scripts/03_pretrain_trajgpt.py` using the updated "
                "`configs/trajgpt_ehrshot.yaml`, then extract embeddings."
            ) from e
        self.model.eval()
        print(f"TrajGPT loaded. Device: {self.device}, Params: {self.model.count_parameters():,}")

    def _resolve_device(self, device_cfg: str) -> str:
        if device_cfg != "auto":
            return device_cfg
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def embedding_dim(self) -> int:
        return self._d_model

    def precompute_patient_tokens(self, patient_data: dict[int, dict]):
        """Pre-encode code tokens once per patient."""
        self._patient_tokens = {}
        self._patient_times = {}
        for pid, seq in patient_data.items():
            self._patient_tokens[pid] = self.tokenizer.encode(seq["codes"])
            self._patient_times[pid] = seq["times"]

    def _prepare_patient(self, subject_id: int, prediction_time: datetime) -> dict | None:
        if subject_id not in self._patient_times:
            return None

        times = self._patient_times[subject_id]
        tokens = self._patient_tokens[subject_id]

        cutoff = bisect.bisect_right(times, prediction_time)
        if cutoff == 0:
            return None

        start = max(0, cutoff - self.max_seq_len)
        seq_tokens = tokens[start:cutoff]
        seq_times = times[start:cutoff]

        t0 = seq_times[0]
        days = []
        for t in seq_times:
            delta = t - t0
            if hasattr(delta, "total_seconds"):
                days.append(delta.total_seconds() / 86400.0)
            else:
                days.append(float(delta) / 1e9 / 86400.0 if hasattr(delta, "__float__") else 0.0)

        return {
            "token_ids": seq_tokens,
            "timestamps": days,
            "length": len(seq_tokens),
        }

    def embed_patients(
        self,
        patient_data: dict[int, dict],
        prediction_times: list[tuple[int, datetime]],
    ) -> np.ndarray:
        all_embeddings = np.zeros((len(prediction_times), self.embedding_dim), dtype=np.float32)

        if not hasattr(self, "_patient_tokens"):
            self.precompute_patient_tokens(patient_data)

        batch_samples = []
        batch_indices = []
        n_valid = 0

        for i, (subject_id, pred_time) in enumerate(
            tqdm(prediction_times, desc="Extracting TrajGPT embeddings", mininterval=5)
        ):
            sample = self._prepare_patient(subject_id, pred_time)
            if sample is None:
                continue

            batch_samples.append(sample)
            batch_indices.append(i)
            n_valid += 1

            if len(batch_samples) >= self.batch_size:
                self._run_batch(batch_samples, batch_indices, all_embeddings)
                batch_samples = []
                batch_indices = []

        if batch_samples:
            self._run_batch(batch_samples, batch_indices, all_embeddings)

        print(f"  {n_valid}/{len(prediction_times)} samples embedded")
        return all_embeddings

    def _run_batch(
        self,
        batch_samples: list[dict],
        batch_indices: list[int],
        all_embeddings: np.ndarray,
    ) -> None:
        max_len = max(s["length"] for s in batch_samples)
        B = len(batch_samples)

        padded_ids = np.zeros((B, max_len), dtype=np.int64)
        padded_ts = np.zeros((B, max_len), dtype=np.float32)
        mask = np.zeros((B, max_len), dtype=bool)

        for j, s in enumerate(batch_samples):
            n = s["length"]
            padded_ids[j, :n] = s["token_ids"]
            padded_ts[j, :n] = s["timestamps"]
            mask[j, :n] = True

        token_ids_t = torch.tensor(padded_ids, dtype=torch.long, device=self.device)
        timestamps_t = torch.tensor(padded_ts, dtype=torch.float32, device=self.device)
        masks_t = torch.tensor(mask, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            representations = self.model.extract_representations(
                token_ids_t,
                timestamps_t,
                masks_t,
                forward_impl="parallel",
            )

        all_embeddings[batch_indices] = representations.cpu().numpy()
