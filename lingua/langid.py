# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger()

GLOTLID_REPO_ID = "cis-lmu/glotlid"
GLOTLID_FILENAME = "model.bin"
_LABEL_PREFIX = "__label__"


@dataclass
class LangIdArgs:
    enabled: bool = False
    model_repo_id: str = GLOTLID_REPO_ID
    model_filename: str = GLOTLID_FILENAME
    model_path: Optional[str] = None  # local override, skips the HF Hub download
    min_confidence: float = 0.0  # predictions below this are treated as unknown (None)
    # Predicted GlotLID code (e.g. "fra_Latn") -> tokenizer key/name, same shape as
    # OracleRoutingArgs.source_to_tokenizer.
    lang_to_tokenizer: Any = field(default_factory=dict)


class GlotLIDLanguageIdentifier:
    """Thin wrapper around a fastText GlotLID model for per-text language prediction.

    GlotLID predicts labels in the `{lang3}_{Script}` format (e.g. `fra_Latn`), which
    matches the keys already used for routing tables such as
    `OracleRoutingArgs.source_to_tokenizer`.
    """

    def __init__(self, args: LangIdArgs):
        import fasttext

        model_path = args.model_path
        if model_path is None:
            import huggingface_hub as hf_hub

            model_path = hf_hub.hf_hub_download(args.model_repo_id, args.model_filename)
            logger.info(f"Downloaded GlotLID model from {args.model_repo_id} to {model_path}")
        self.model = fasttext.load_model(model_path)
        self.min_confidence = args.min_confidence
        # If a routing table is configured, only ever route to a language that's
        # actually in the mix rather than the model's raw top-1 over ~2000 languages.
        self.allowed_langs = set(args.lang_to_tokenizer) if args.lang_to_tokenizer else None
        self._num_labels = len(self.model.get_labels())

    def predict(self, text: str) -> Optional[str]:
        text = text.replace("\n", " ").strip()
        if not text:
            return None
        # Call the pybind predict directly instead of fasttext's Python wrapper:
        # that wrapper does `np.array(probs, copy=False)`, which raises on numpy>=2.0.
        k = self._num_labels if self.allowed_langs else 1
        predictions = self.model.f.predict(text, k, 0.0, "strict")
        if not predictions:
            return None
        if self.allowed_langs:
            predictions = [
                (prob, label) for prob, label in predictions
                if label[len(_LABEL_PREFIX):] in self.allowed_langs
            ]
            if not predictions:
                return None
            # fastText's probs are already a softmax over all ~2000 labels, so
            # dividing by the subset sum is equivalent to re-softmaxing restricted
            # to the allowed languages: P(y|x, y in S) = P(y|x) / sum_{y' in S} P(y'|x).
            total = sum(prob for prob, _ in predictions)
            if total <= 0:
                return None
            predictions = [(prob / total, label) for prob, label in predictions]
        prob, label = predictions[0]
        if float(prob) < self.min_confidence:
            return None
        return label[len(_LABEL_PREFIX):]

    def predict_batch(self, texts: List[str]) -> List[Optional[str]]:
        labels = [self.predict(t) for t in texts]
        if any(l is None for l in labels):
            logger.warning(f"{sum(l is None for l in labels)} / {len(labels)} texts were below the min_confidence threshold of {self.min_confidence} and will be treated as unknown, i.e. randomly routed.")
            logger.warning(f"Some examples: {[(t, l) for t, l in zip(texts, labels) if l is None][:5]}")
        return labels
