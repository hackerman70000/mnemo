from __future__ import annotations

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFModel:
    name: str

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.name = model_name
        self.device = self._resolve_device(device)
        self.dtype = dtype or self._default_dtype(self.device)

        logger.info(f"Loading {model_name} on {self.device} ({self.dtype})")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=self.dtype,
            trust_remote_code=trust_remote_code,
        )
        model.to(torch.device(self.device))  # type: ignore[arg-type]
        model.eval()  # type: ignore[no-untyped-call]
        self.model = model

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _default_dtype(device: str) -> torch.dtype:
        return torch.float16 if device in {"cuda", "mps"} else torch.float32

    @torch.no_grad()
    def token_logprobs(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"][0]

        if input_ids.size(0) < 2:
            return np.array([], dtype=np.float32)

        logits = self.model(**inputs).logits[0]
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        targets = input_ids[1:].unsqueeze(-1)
        token_log_probs = log_probs[:-1].gather(-1, targets).squeeze(-1)
        return token_log_probs.cpu().numpy()
