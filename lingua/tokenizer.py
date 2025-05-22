# Copyright (c) Meta Platforms, Inc. and affiliates.

import abc
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Sequence
import logging
import os

from sentencepiece import SentencePieceProcessor
import tiktoken
from tiktoken.load import load_tiktoken_bpe

logger = logging.getLogger(__name__)


@dataclass
class TokenizerArgs:
    name: str = "bytes"
    path: Optional[str] = None


class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def encode(self, tokens, add_bos, add_eos):
        pass

    @abc.abstractmethod
    def decode(self, tokens):
        pass

    @abc.abstractmethod
    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        """Return the offsets of the tokens in the original text. Only used for evaluation."""
        pass


class MockTokenizer(Tokenizer):
    n_words: int = 256

    def encode(self, tokens, add_bos, add_eos):
        return tokens


class ByteTokenizer(Tokenizer):
    def __init__(self):
        self.bos_id = 256
        self.eos_id = 257
        self.n_words = 258

    def encode(self, s: str, add_bos: bool = False, add_eos: bool = False):
        tokens = [self.bos_id] * add_bos + list(s.encode()) + [self.eos_id] * add_eos
        return tokens

    def decode(self, tokens: List[int]):
        byte_tokens = bytes([t for t in tokens if t < 256])
        return byte_tokens.decode("utf-8", errors="backslashreplace")

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        if tokens is None:
            tokens = self.encode(text)

        decoded_chars, offsets = [], []
        byte_pos = 0
        for token in tokens:
            if token < 256:
                char = bytes([token]).decode("utf-8", errors="ignore")
                if char:
                    decoded_chars.append(char)
                    offsets.append(byte_pos)
                byte_pos += len(char.encode("utf-8"))

        return decoded_chars, offsets


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, model_path: str) -> None:
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, add_bos: bool, add_eos: bool):
        assert type(s) is str
        tokens = (
            [self.bos_id] * add_bos + self.sp_model.encode(s) + [self.eos_id] * add_eos
        )
        return tokens

    def decode(self, tokens: List[int]):
        return self.sp_model.decode(tokens)

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        pieces = self.sp_model.encode_as_immutable_proto(text).pieces
        substrs = [p.surface for p in pieces]
        offsets = [p.begin for p in pieces]
        return substrs, offsets


DEFAULT_TIKTOKEN_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
DEFAULT_TIKTOKEN_SPECIAL_TOKENS = {
    "<|begin_of_text|>": 0,
    "<|end_of_text|>": 1,
    "<|fim_prefix|>": 2,
    "<|fim_middle|>": 3,
    "<|fim_end_fill|>": 253,
    "<|fim_pad|>": 254,
    "<|fim_suffix|>": 255,
}
TIKTOKEN_MAX_ENCODE_CHARS = 400_000


class TikTokenTokenizer(Tokenizer):

    def __init__(self, model_path: str) -> None:
        mergeable_ranks = load_tiktoken_bpe(model_path)
        all_special_tokens_with_ids = copy(DEFAULT_TIKTOKEN_SPECIAL_TOKENS)
        missing_ids = set(range(256)) - set(all_special_tokens_with_ids.values())
        for id in missing_ids:
            all_special_tokens_with_ids[f"<|reserved_special_token_{id}|>"] = id
        for name in all_special_tokens_with_ids:
            all_special_tokens_with_ids[name] += len(mergeable_ranks)

        self.tkt_model = tiktoken.core.Encoding(
            name=Path(model_path).stem,
            pat_str=DEFAULT_TIKTOKEN_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=all_special_tokens_with_ids,
        )

        self.bos_id: int = self.tkt_model.encode_single_token("<|begin_of_text|>")
        self.eos_id: int = self.tkt_model.encode_single_token("<|end_of_text|>")

        self.n_words: int = self.tkt_model.n_vocab

        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    def encode(self, s: str, add_bos: bool, add_eos: bool):
        assert isinstance(s, str)

        subs = []
        for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS):
            subs.append(s[i : i + TIKTOKEN_MAX_ENCODE_CHARS])
        return (
            [self.bos_id] * add_bos
            + sum(self.tkt_model.encode_ordinary_batch(subs), start=[])
            + [self.eos_id] * add_eos
        )

    def decode(self, tokens: List[int]):
        return self.tkt_model.decode(tokens)

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        if tokens is not None:
            token_bytes = self.tkt_model.decode_tokens_bytes(tokens)
        else:
            token_bytes = self.tkt_model.decode_tokens_bytes(
                self.tkt_model.encode(text, allowed_special="all")
            )

        text_len, offsets = 0, []
        for token in token_bytes:
            offsets.append(max(0, text_len - (0x80 <= token[0] < 0xC0)))
            text_len += sum(1 for c in token if not 0x80 <= c < 0xC0)
        substrs = [text[s:e] for s, e in zip(offsets, offsets[1:] + [None])]
        return substrs, offsets


def find_id(tokenizer, surfaces: Sequence[str]):
    """Look through surfaces to see if any are in the tokenizer's vocab."""
    token_id = None
    for surface in surfaces:
        token_id = tokenizer.token_to_id(surface)
        if token_id is not None:
            logger.info("Found id for special token: %s", surface)
            break
    else:
        logger.warning("No id found for special token.")
    return token_id


class HFTokenizer(Tokenizer):

    def __init__(self, model_path: str) -> None:
        try:
            import transformers
            # Try to load as a transformers.Tokenizer as it includes more
            # information about things like bos/eos
            transformers_tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_path
            )
            logger.info("Loaded Transformers Tokenizer from %s", model_path)
            # Extract the underlying tokenizers.Tokenizer to get access to things
            # like the offests.
            self.hf_tokenizer = transformers_tokenizer._tokenizer
            logger.info(
                "Extracted Tokenizers Tokenizer from Transformers Tokenizer"
            )

            # Find special tokens based on the transformers.Tokenizer
            bos_token = transformers_tokenizer.bos_token
            logger.info(
                "Found bos_token: %s based on Transformers Tokenizer.", bos_token
            )
            self.bos_id = transformers_tokenizer.convert_tokens_to_ids(bos_token)

            eos_token = transformers_tokenizer.eos_token
            logger.info(
                "Found eos_token: %s based on Transformers Tokenizer.", eos_token
            )
            self.eos_id = transformers_tokenizer.convert_tokens_to_ids(eos_token)

            pad_token = transformers_tokenizer.pad_token
            logger.info(
                "Found pad_token: %s based on Transformers Tokenizer.", pad_token
            )
            if pad_token is not None:
                # It is ok for this not be set for models that don't have a pad
                # because it isn't set for some the other lingua implementations.
                self.pad_id = transformers_tokenizer.convert_tokens_to_ids(pad_token)

        except:
            import tokenizers
            # If we failed to load as a transformers.Tokenizer, load as a
            # tokenizers.Tokenizer
            self.hf_tokenizer = tokenizers.Tokenizer.from_file(model_path)
            logger.info("Loaded Tokenizers Tokenizer.")

            # We need to infer the special tokens. If you used a different
            # special token, it needs to be added tothe DEFAULT_SPECIAL_TOKENS
            # dict.
            logger.info("Infering bos id.")
            self.bos_id = find_id(self.hf_tokenizer, DEFAULT_SPECIAL_TOKENS["bos"])
            logger.info("Infering eos id.")
            self.eos_id = find_id(self.hf_tokenizer, DEFAULT_SPECIAL_TOKENS["eos"])
            logger.info("Infering pad id.")
            self.pad_id = find_id(self.hf_tokenizer, DEFAULT_SPECIAL_TOKENS["pad"])

        self.n_words = self.hf_tokenizer.get_vocab_size()

        logger.info(
            "#words: %d - BOS ID: %d - EOS ID: %d",
            self.n_words,
            self.bos_id,
            self.eos_id,
        )

    def encode(self, s: str, add_bos: bool, add_eos: bool):
        """Convert a string to a list of tokens."""
        # Never add bos/eos special tokens because we are using a
        # tokenizers.Tokenizer which doesn't auto add them.
        encoded = self.hf_tokenizer.encode(s, add_special_tokens=False).ids
        # Add bos/eos as needed, easy because we are not processing batches.
        if add_bos:
            encoded = [self.bos_id] + encoded
        if add_eos:
            encoded = encoded + [self.eos_id]
        return encoded

    def decode(self, tokens: List[int]):
        """Convert a list of tokens to a stirng."""
        return self.hf_tokenizer.decode(tokens)

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        """Get the offsets (and surface) for each token in the original string."""
        if tokens is not None:
            logger.warning(
                "`tokens` passed to `get_token_offsets`, but are ignored with the HFTokenizer."
            )

        # Don't add special tokens so we don't need to handle things like the
        # offset of the bos token.
        encoding = self.hf_tokenizer.encode(text, add_special_tokens=False)
        # Slice the original text instead of using encoding.tokens to avoid the
        # fact that tokenizers uses Ġ instead of space.
        substrs = [text[s:e] for s, e in encoding.offsets]
        return substrs, encoding.offsets


class TokenMonsterTokenizer(Tokenizer):

    def __init__(self, model_path: str):
        import tokenmonster
        self.tokenizer = tokenmonster.load(model_path)
        self.n_words = self.tokenizer.vocab_size
        self.bos_id = None
        self.eos_id = None

        logger.info(
            "#words: %d - BOS ID: %d - EOS ID: %d",
            self.n_words,
            self.bos_id,
            self.eos_id,
        )

    def encode(self, s: str, add_bos: bool, add_eos: bool):
        return self.tokenizer.tokenize(s)

    def decode(self, tokens: List[int]):
        return self.tokenizer.decode(tokens)

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        return None, None


class TekkenTokenizer(Tokenizer):
    def __init__(self):
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        tok = MistralTokenizer.v3(is_tekken=True)
        self.tokenizer = tok.instruct_tokenizer.tokenizer

        self.n_words = self.tokenizer.n_words
        self.bos_id = self.tokenizer.bos_id
        self.eos_id = self.tokenizer.eos_id

        logger.info(
            "#words: %d - BOS ID: %d - EOS ID: %d",
            self.n_words,
            self.bos_id,
            self.eos_id,
        )

    def encode(self, s: str, add_bos: bool, add_eos: bool):
        return self.tokenizer.encode(s, add_bos, add_eos)

    def decode(self, tokens: List[int]):
        if tokens[0] == self.bos_id:
            tokens = tokens[1:]
        if tokens[-1] == self.eos_id:
            tokens = tokens[:-1]
        return self.tokenizer.decode(tokens)

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        return None, None


def build_tokenizer(name: str, path: Optional[str] = None) -> Tokenizer:
    if name == "bytes":
        return ByteTokenizer()
    elif name == "mock":
        return MockTokenizer()
    elif name == "sp":
        return SentencePieceTokenizer(path)
    elif name == "tiktoken":
        return TikTokenTokenizer(path)
    elif name == "huggingface":
        return HFTokenizer(path)
    elif name == "tokenmonster":
        return TokenMonsterTokenizer(path)
    elif name == "tekken":
        return TekkenTokenizer()
    else:
        raise NotImplementedError(f"{name} tokenizer type is not implemented")
