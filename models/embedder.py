from __future__ import annotations

"""Patient embedding extraction.

Defines the PatientEmbedder ABC and the CLMBRBaseEmbedder implementation
that wraps Stanford's FEMR library for CLMBR-T-base.
"""

from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm


class PatientEmbedder(ABC):
    """Abstract base class for patient embedding models.

    Subclasses must implement embed_patients(), which takes patient data
    and prediction times, and returns fixed-size embedding vectors.
    """

    @abstractmethod
    def embed_patients(
        self,
        patient_data: dict[int, dict],
        prediction_times: list[tuple[int, datetime]],
    ) -> np.ndarray:
        """Extract embeddings for patients at specific prediction times.

        Args:
            patient_data: Dict mapping subject_id -> {
                "codes": list[str], "times": list[datetime], ...
            }
            prediction_times: List of (subject_id, prediction_time) tuples.

        Returns:
            Embedding matrix of shape (len(prediction_times), embedding_dim).
        """
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output embeddings."""
        ...


class CLMBRBaseEmbedder(PatientEmbedder):
    """CLMBR-T-base embedder using Stanford's FEMR library.

    Loads the pretrained 141M parameter model from HuggingFace and
    extracts 768-dim patient representations.
    """

    def __init__(
        self,
        model_hub_id: str = "StanfordShahLab/clmbr-t-base",
        device: str = "cpu",
        batch_size: int = 64,
    ):
        import femr.models.tokenizer
        import femr.models.processor
        import femr.models.transformer

        self.device = device
        self.batch_size = batch_size

        print(f"Loading CLMBR-T-base from {model_hub_id}...")
        self.tokenizer = femr.models.tokenizer.FEMRTokenizer.from_pretrained(
            model_hub_id
        )
        self.processor = femr.models.processor.FEMRBatchProcessor(self.tokenizer)
        self.model = femr.models.transformer.FEMRModel.from_pretrained(model_hub_id)
        self.model = self.model.to(device)
        self.model.eval()
        print(f"CLMBR-T-base loaded. Device: {device}")

    @property
    def embedding_dim(self) -> int:
        return 768

    def _build_patient_for_femr(self, patient_seq: dict, prediction_time: datetime):
        """Convert our patient dict format to what FEMR expects.

        FEMR's processor.convert_patient expects a patient object with events.
        The exact format depends on the FEMR version. This method handles
        the conversion and truncation at prediction_time.
        """
        codes = patient_seq["codes"]
        times = patient_seq["times"]

        # Truncate at prediction_time (no future leakage)
        truncated_codes = []
        truncated_times = []
        for code, time in zip(codes, times):
            if time <= prediction_time:
                truncated_codes.append(code)
                truncated_times.append(time)

        return {
            "codes": truncated_codes,
            "times": truncated_times,
            "patient_id": 0,  # placeholder
        }

    def embed_patients(
        self,
        patient_data: dict[int, dict],
        prediction_times: list[tuple[int, datetime]],
    ) -> np.ndarray:
        """Extract CLMBR-T-base embeddings.

        Uses FEMR's batch processor for tokenization and the pretrained
        Transformer for inference. Truncates each patient's timeline at
        the prediction_time to prevent information leakage.
        """
        embeddings = []
        batch_raw = []
        batch_indices = []

        for i, (subject_id, pred_time) in enumerate(
            tqdm(prediction_times, desc="Extracting CLMBR embeddings")
        ):
            if subject_id not in patient_data:
                embeddings.append(np.zeros(self.embedding_dim))
                continue

            patient_seq = patient_data[subject_id]
            patient = self._build_patient_for_femr(patient_seq, pred_time)

            try:
                raw_batch = self.processor.convert_patient(patient, tensor_type="pt")
                batch_raw.append(raw_batch)
                batch_indices.append(i)
            except Exception as e:
                print(f"Warning: Failed to process patient {subject_id}: {e}")
                embeddings.append(np.zeros(self.embedding_dim))
                continue

            # Process batch when full
            if len(batch_raw) >= self.batch_size:
                batch_emb = self._process_batch(batch_raw)
                for idx, emb in zip(batch_indices, batch_emb):
                    while len(embeddings) <= idx:
                        embeddings.append(np.zeros(self.embedding_dim))
                    embeddings[idx] = emb
                batch_raw = []
                batch_indices = []

        # Process remaining
        if batch_raw:
            batch_emb = self._process_batch(batch_raw)
            for idx, emb in zip(batch_indices, batch_emb):
                while len(embeddings) <= idx:
                    embeddings.append(np.zeros(self.embedding_dim))
                embeddings[idx] = emb

        # Ensure correct length
        while len(embeddings) < len(prediction_times):
            embeddings.append(np.zeros(self.embedding_dim))

        return np.array(embeddings[: len(prediction_times)])

    def _process_batch(self, batch_raw: list) -> list[np.ndarray]:
        """Run a batch through the model and extract representations."""
        batch = self.processor.collate(batch_raw)

        # Move tensors to device
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with torch.no_grad():
            _, result = self.model(**batch)
            representations = result["representations"]

        return [
            representations[i].cpu().numpy() for i in range(representations.shape[0])
        ]
