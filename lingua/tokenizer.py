# Copyright (c) Meta Platforms, Inc. and affiliates.

import abc
import logging
import os
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tiktoken
from sentencepiece import SentencePieceProcessor
from tiktoken.load import load_tiktoken_bpe

logger = logging.getLogger(__name__)


@dataclass
class TokenizerArgs:
    name: str = "" #"bytes"
    path: Optional[str] = None
    tokenizers: Optional[List[Dict[str, Any]]] = None
    load_supermapping: Optional[bool] = False
    dropout: float = 0.0
    # following are needed for supertokenizer
    seed: Optional[int] = 42
    superset_code_name: Optional[str] = "super_vocab"
    n_words: Optional[int] = None 


class Tokenizer(abc.ABC):
    mapping: Dict[str, Any] = {}

    @abc.abstractmethod
    def encode(self, tokens, add_bos, add_eos):
        pass

    @abc.abstractmethod
    def decode(self, tokens,skip_special_tokens:bool=True):
        pass

    @abc.abstractmethod
    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        """Return the offsets of the tokens in the original text. Only used for evaluation."""
        pass

    def load_supermapping(self, base_path: str, path: str, superset_code_name: str = "super_vocab") -> None:
        import json

        mapping_path = Path(base_path, f"{path.replace('/', '--')}_super_mapping.json")
        if not mapping_path.exists():
            import huggingface_hub as hf_hub

            try:
                assert os.environ.get("HF_HUB_OFFLINE") != "1"
                repo_id = path
                # e.g. fineweb2_hq/flexitok--bpe_arb_Arab_8000_super_mapping.json
                # import code; code.interact(local=locals()|globals())
                # 8k_v2/flexitok--bpe_ltr_vie_Latn_8000_v2_super_mapping.json
                mapping_path = hf_hub.hf_hub_download(
                    repo_id, f"{superset_code_name}/{path.replace('/', '--')}_super_mapping.json"
                )
                logger.info(f"Downloaded super mapping from HF Hub {repo_id} to {mapping_path}")
            # except hf_hub.errors.file
            ## backward compatibility for old path format
            except hf_hub.errors.RepositoryNotFoundError as e:
                assert os.environ.get("HF_HUB_OFFLINE") != "1"
                repo_id = f"gsaltintas/supertokenizer-{path.replace('/', '-')}"
                mapping_path = hf_hub.hf_hub_download(
                    repo_id, f"{path.replace('/', '--')}_super_mapping.json"
                )
                logger.info(f"Downloaded super mapping from HF Hub {repo_id} to {mapping_path}")
            except hf_hub.errors.RepositoryNotFoundError as e:
                assert os.environ.get("HF_HUB_OFFLINE") != "1"
                repo_id = f"flexitok/supertokenizer-{path.replace('/', '-')}"
                mapping_path = hf_hub.hf_hub_download(
                    repo_id, f"{path.replace('/', '--')}_super_mapping.json"
                )
                logger.info(f"Downloaded super mapping from HF Hub {repo_id} to {mapping_path}")
            except:
                mapping_path = Path(base_path, f"{path.replace('/', '--')}_super_mapping.json")
                logger.warning(f"Failed to download super mapping from HF Hub {repo_id}. Trying local path {mapping_path}")
        assert os.path.isfile(mapping_path), mapping_path
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
        self.mapping = {int(k): v for k, v in mapping.items()}
        logger.info(f"Loaded super mapping from {mapping_path}")

    def encode_to_supermapping(self, tokens: List[str], add_bos: bool, add_eos: bool) -> List[int]:
        ids = []
        token_ids = self.encode(tokens, add_bos=add_bos, add_eos=add_eos)
        if len(self.mapping) == 0:
            return token_ids
        return [self.mapping[tid] for tid in token_ids if tid in self.mapping ]

    def decode_from_supermapping(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        if len(self.mapping) == 0:
            return self.decode(tokens, skip_special_tokens=skip_special_tokens)
        reverse_mapping = {v: k for k, v in self.mapping.items()}
        token_ids = [reverse_mapping[tid] for tid in tokens if tid in reverse_mapping]
        return self.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
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

    def decode(self, tokens: List[int],skip_special_tokens:bool=True):
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
    def __init__(self, model_path: str, alpha: float=0.0) -> None:
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.alpha = alpha

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
        # s.encode('New York', out_type=str, enable_sampling=True, alpha=0.1, nbest_size=-1)
        tokens = (
            [self.bos_id] * add_bos 
            + self.sp_model.encode(s, enable_sampling=self.alpha>0, alpha=self.alpha, nbest_size=-1) 
            + [self.eos_id] * add_eos
        )
        return tokens

    def decode(self, tokens: List[int],skip_special_tokens:bool=True):
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
DEFAULT_SPECIAL_TOKENS = {
    "bos": ["<|begin_of_text|>", "<s>", "<bos>"],
    "eos": ["<|end_of_text|>", "</s>", "<eos>"],
    "pad": ["<pad>"],
}
ALIGNED_BOS = "~SPECIAL~ALIGNED~BOS~SYMBOL~"

class TikTokenTokenizer(Tokenizer):

    def __init__(self, model_path: str) -> None:
        try:
            # on Vulcan need to first load these because there is no internet connection on the compute nodes
            self.tkt_model = tiktoken.encoding_for_model(model_path)
        except:
            mergeable_ranks = load_tiktoken_bpe(model_path)
            all_special_tokens_with_ids = copy(DEFAULT_TIKTOKEN_SPECIAL_TOKENS)
            missing_ids = set(range(256)) - set(all_special_tokens_with_ids.values())
            for id in missing_ids:
                all_special_tokens_with_ids[f"<|reserved_special_token_{id}|>"] = id
            for name in all_special_tokens_with_ids:
                all_special_tokens_with_ids[name] += len(mergeable_ranks)
            logger.error(f"Failed to load TikToken model from {model_path}")
            mergeable_ranks = load_tiktoken_bpe(model_path)
            self.tkt_model = tiktoken.core.Encoding(
                name=Path(model_path).stem,
                pat_str=DEFAULT_TIKTOKEN_PATTERN,
                mergeable_ranks=mergeable_ranks,
                special_tokens=all_special_tokens_with_ids,
            )

        try:
            self.bos_id: int = self.tkt_model.encode_single_token("<|begin_of_text|>")
        except:
            self.bos_id: int = None
        self.eos_id: int = self.tkt_model.encode_single_token("<|endoftext|>")

        self.n_words: int = self.tkt_model.n_vocab

        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    def encode(self, s: str, add_bos: bool, add_eos: bool):
        assert isinstance(s, str)
        
        add_bos = self.bos_id is not None and add_bos

        subs = []
        for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS):
            subs.append(s[i : i + TIKTOKEN_MAX_ENCODE_CHARS])
        return (
            [self.bos_id] * add_bos
            + sum(self.tkt_model.encode_ordinary_batch(subs), start=[])
            + [self.eos_id] * add_eos
        )

    def decode(self, tokens: List[int],skip_special_tokens:bool=False):
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

    def __init__(self, model_path: str, dropout: float = 0) -> None:
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

            if dropout > 0:
                try:
                    self.hf_tokenizer.model.dropout = dropout
                    logger.info("Set tokenizer dropout to %f", dropout)
                except Exception as e:
                    logger.warning("Failed to set tokenizer dropout: %s", e)

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
            if dropout > 0:
                try:
                    self.hf_tokenizer._tokenizer.model.dropout = dropout
                    logger.info("Set tokenizer dropout to %f", dropout)
                except Exception as e:
                    logger.warning("Failed to set tokenizer dropout: %s", e)

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
        if add_bos and self.bos_id is not None:
            encoded = [self.bos_id] + encoded
        if add_eos and self.eos_id is not None:
            encoded = encoded + [self.eos_id]
        return encoded

    def decode(self, tokens: List[int], skip_special_tokens: bool = True):
        """Convert a list of tokens to a stirng."""
        return self.hf_tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

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


class ByT5HFTokenizer(HFTokenizer):

    def __init__(self, model_path: str) -> None:
        import transformers
        self.hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.bos_token = self.hf_tokenizer.pad_token
        self.eos_token = self.hf_tokenizer.eos_token
        self.bos_id = self.hf_tokenizer.convert_tokens_to_ids(self.bos_token)
        self.eos_id = self.hf_tokenizer.convert_tokens_to_ids(self.eos_token)

        self.n_words = self.hf_tokenizer.vocab_size

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
        encoded = self.hf_tokenizer.encode(s, add_special_tokens=False)
        # Add bos/eos as needed, easy because we are not processing batches.
        if add_bos and self.bos_id is not None:
            encoded = [self.bos_id] + encoded
        if add_eos and self.eos_id is not None:
            encoded = encoded + [self.eos_id]
        return encoded

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        """Get the offsets (and surface) for each token in the original string."""
        return None, None


class SimplifiedHFTokenizer(HFTokenizer):

    def __init__(self, model_path: str, dropout: float = 0) -> None:
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
        if dropout > 0:
            try:
                self.hf_tokenizer.model.dropout = dropout
                logger.info("Set tokenizer dropout to %f", dropout)
            except Exception as e:
                logger.warning("Failed to set tokenizer dropout: %s", e)
        special_tokens = getattr(transformers_tokenizer, "special_tokens_map", {})
        if "bert" in model_path:
            self.bos_token = special_tokens.get("cls_token")
        elif "t5" in model_path:
            self.bos_token = special_tokens.get("pad_token")
        else:
            self.bos_token = special_tokens.get("bos_token")
        logger.info(
            "Found bos_token: %s based on Transformers Tokenizer.", self.bos_token
        )
        if self.bos_token is not None:
            self.bos_id = transformers_tokenizer.convert_tokens_to_ids(self.bos_token)
        else:
            self.bos_id = None
        logger.info(
            "Found bos_id: %s based on Transformers Tokenizer.", self.bos_id
        )

        if "bert" in model_path:
            self.eos_token = special_tokens.get("pad_token")
        else:
            self.eos_token = special_tokens.get("eos_token")
        logger.info(
            "Found eos_token: %s based on Transformers Tokenizer.", self.eos_token
        )
        if self.eos_token is not None:
            self.eos_id = transformers_tokenizer.convert_tokens_to_ids(self.eos_token)
        else:
            self.eos_id = None
        logger.info(
            "Found eos_id: %s based on Transformers Tokenizer.", self.eos_id
        )

        self.n_words = self.hf_tokenizer.get_vocab_size()

        logger.info(
            "#words: %d - BOS ID: %d - EOS ID: %d",
            self.n_words,
            self.bos_id,
            self.eos_id,
        )


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
        token_ids = self.tokenizer.tokenize(s)
        if token_ids is None:
            return np.array([], dtype=np.longlong)
        return token_ids.astype(np.longlong)

    def decode(self, tokens: List[int], skip_special_tokens: bool = True):
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

    def decode(self, tokens: List[int], skip_special_tokens: bool = True):
        if len(tokens) == 0:
            return ""
        if skip_special_tokens and tokens[0] == self.bos_id :
            tokens = tokens[1:]
        if skip_special_tokens and tokens[-1] == self.eos_id:
            tokens = tokens[:-1]
        return self.tokenizer.decode(tokens)

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        return None, None

class SupersetTokenizer(Tokenizer):
    n_words: int = 851586
    def __init__(self, tokenizers: List[Dict[str, str]], rng_state: Dict[str, Any] = None, superset_code_name: str = "super_vocab", n_words: Optional[int] = None):
        self.tokenizers = {}
        self.superset_code_name = superset_code_name
        self.n_words = n_words if n_words is not None else self.n_words
        ## todo: need to load mappings too
        import os
        for tokenizer_info in tokenizers:
            name = tokenizer_info["name"]
            path = tokenizer_info.get("path", None)
            kwargs = {}
            dropout = tokenizer_info.get("dropout", 0)
            if dropout > 0:
                kwargs["dropout"] = dropout
            load_supermapping = tokenizer_info.get("load_supermapping", False)
            try:
                tokenizer = build_tokenizer(name, path, **kwargs)
                encoding_path = path
                if name == "tiktoken":
                    encoding_path = f"tiktoken/{path}"
                elif name == "tokenmonster":
                    encoding_path = f"tokenmonster/{path}"
                elif name == "tekken":
                    encoding_path = f"mistralai/{path}"
                if load_supermapping:
                    logger.info(f"Loading supermapping for the tokenizer {path}")
                    tokenizer.load_supermapping(f"{os.environ.get('PROJECT')}/tokenizers/super_mappings", encoding_path, superset_code_name)
                else:
                    logger.info(f"Not loading supermapping for the tokenizer {path}")
                self.tokenizers[f"{name}/{path}"] = tokenizer
                logger.info(f"Loaded tokenizer {name} from {path}")
            except Exception as e:
                logger.error("Error loading tokenizer %s from  %s. %s",path, name, e)
        if len(self.tokenizers) == 0:
            raise ValueError("No valid tokenizers provided.")
        
        logger.info(f"Number of tokenizers loaded: {len(self.tokenizers)}")
        if rng_state is not None:
            # import code; code.interact(local=dict(globals(), **locals()))
            if isinstance(rng_state, int):
                rng = np.random.default_rng(seed=rng_state)
            else:
                rng = np.random.default_rng()
                rng.bit_generator.state = rng_state
            self.rng = rng
            logger.info("Restored RNG state for supertokenizer.")
        else:
            self.rng = np.random.default_rng(seed=42)
            logger.info("Initialized new RNG for supertokenizer.")
        import json

        import huggingface_hub as hf_hub
        path = f"{os.environ.get('PROJECT')}/tokenizers/{superset_code_name}/super_vocab.json"
        if not Path(path).exists():
            try:
                supervocab_repo_id = f"flexitok/supertokenizer-{superset_code_name}"
                # flexitok/supertokenizer-fineweb2_hq
                assert os.environ.get("HF_HUB_OFFLINE") != "1"
                repo_id = supervocab_repo_id
                path = hf_hub.hf_hub_download(repo_id, "super_vocab.json")
                logger.info(f"Downloaded super mapping from HF Hub {repo_id} to {path}")
            ## backward compatibility for old path format
            except hf_hub.errors.RepositoryNotFoundError as e:
                assert os.environ.get("HF_HUB_OFFLINE") != "1"
                repo_id = "gsaltintas/supertokenizer-super_vocab"
                path = hf_hub.hf_hub_download(repo_id, "super_vocab.json")
            except hf_hub.errors.RepositoryNotFoundError as e:
                raise ValueError(f"Failed to download super vocab from HF Hub. Tried repo_id {repo_id}. Error: {e}")
        with open(path, "r") as f:
            self.super_vocab = json.load(f)
        # align bos eos with llama
        self.bos_id = self.super_vocab.get(ALIGNED_BOS)
        self.eos_id = self.super_vocab.get("<|end_of_text|>")
        self.bos_token, self.eos_token = ALIGNED_BOS, "<|end_of_text|>"
        if self.eos_id is None:
            self.eos_id = self.super_vocab.get("</s>")
            self.bos_token, self.eos_token = ALIGNED_BOS, "</s>"
            
        logger.info(
            "Setting bos_token: %s with id %d.", self.bos_token, self.bos_id
        )
        logger.info(
            "Setting eos_token: %s with id %d.", self.eos_token, self.eos_id
        )
    def _resolve_tokenizer_choice(self, tokenizer_choice: Optional[Union[int, str]]) -> Optional[int]:
        if tokenizer_choice is None:
            return None

        tokenizer_keys = list(self.tokenizers.keys())

        if isinstance(tokenizer_choice, int):
            if tokenizer_choice < 0 or tokenizer_choice >= len(tokenizer_keys):
                raise ValueError(
                    f"tokenizer_choice {tokenizer_choice} is out of range for available tokenizers {tokenizer_keys}"
                )
            return tokenizer_choice

        if isinstance(tokenizer_choice, str):
            if tokenizer_choice in self.tokenizers:
                return tokenizer_keys.index(tokenizer_choice)

            lowered_choice = tokenizer_choice.strip().lower()
            for index, key in enumerate(tokenizer_keys):
                lowered_key = key.lower()
                if lowered_key == lowered_choice:
                    return index
                if lowered_key.endswith(f"/{lowered_choice}"):
                    return index

            return None

        raise TypeError(
            f"tokenizer_choice must be int, str or None, got {type(tokenizer_choice).__name__}"
        )

    def sample_tokenizer(
        self,
        preferred_tokenizer: Optional[Union[int, str]] = None,
        preferred_probability: float = 0.0,
    ):
        tokenizer_keys = list(self.tokenizers.keys())
        if preferred_tokenizer is not None:
            assert preferred_tokenizer in tokenizer_keys, f"Preferred tokenizer {preferred_tokenizer} not in available tokenizers {tokenizer_keys}"
        preferred_probability = float(preferred_probability)
        preferred_probability = max(0.0, min(1.0, preferred_probability))
        preferred_choice = self._resolve_tokenizer_choice(preferred_tokenizer)

        if preferred_choice is not None and self.rng.random() < preferred_probability:
            tokenizer_choice = preferred_choice
        else:
            tokenizer_choice = int(self.rng.choice(len(tokenizer_keys)))
        return tokenizer_choice, tokenizer_keys[tokenizer_choice]
    
    def encode(self, tokens, add_bos, add_eos, tokenizer_choice: Optional[Union[int, str]] = None):
        resolved_choice = self._resolve_tokenizer_choice(tokenizer_choice)
        if resolved_choice is None:
            resolved_choice, tokenizer_key = self.sample_tokenizer()
        else:
            tokenizer_keys = list(self.tokenizers.keys())
            tokenizer_key = tokenizer_keys[resolved_choice]
        tokenizer = self.tokenizers[tokenizer_key]
        ids = tokenizer.encode_to_supermapping(tokens, add_bos=False, add_eos=False)   
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        # logger.debug(f"Selected tokenizer {tokenizer_key}, for string ({tokens[:40]}) length of ids: {len(ids)}, add_bos: {add_bos}, add_eos: {add_eos}")
        return ids

    def decode(self, tokens: List[int],skip_special_tokens:bool=True, tokenizer_choice: Optional[Union[int, str]] = None):
        resolved_choice = self._resolve_tokenizer_choice(tokenizer_choice)
        if resolved_choice is None:
            resolved_choice, tokenizer_key = self.sample_tokenizer()
        else:
            tokenizer_keys = list(self.tokenizers.keys())
            tokenizer_key = tokenizer_keys[resolved_choice]
        tokenizer = self.tokenizers[tokenizer_key]
        return tokenizer.decode_from_supermapping(tokens, skip_special_tokens=skip_special_tokens)
        pass

    def get_token_offsets(self, text: str, tokens: List[int] | None = None) -> Tuple[List[str] | List[int]]:
        return None, None

class RoutedSuperTokenizer(SupersetTokenizer):
    """
    SupersetTokenizer that uses a learned ByteRouter to select the tokenizer.

    The ByteRouter (an nn.Module living on the model's device) is stored here so
    it is easy to load/save alongside the main model checkpoint.  During data
    preprocessing the router is not available; use the parent class's
    sample_tokenizer() instead.  At inference time call encode() normally and
    the router will decide which tokenizer to use based on the raw bytes.

    Args:
        tokenizers:         same as SupersetTokenizer
        router_args:        ByteRouterArgs (imported lazily to avoid a hard torch dep)
        router_checkpoint:  optional path to a standalone router state-dict (.pt)
        **kwargs:           forwarded to SupersetTokenizer
    """

    def __init__(self, tokenizers, router_args=None, router_checkpoint: Optional[str] = None, **kwargs):
        import torch
        super().__init__(tokenizers, **kwargs)
        from lingua.router import ByteRouter, ByteRouterArgs

        tokenizer_names = list(self.tokenizers.keys())
        if router_args is None:
            router_args = ByteRouterArgs(n_tokenizers=len(self.tokenizers), tokenizer_names=tokenizer_names)
        else:
            if router_args.n_tokenizers != len(self.tokenizers):
                logger.warning(
                    "router_args.n_tokenizers (%d) != number of loaded tokenizers (%d); "
                    "overriding.", router_args.n_tokenizers, len(self.tokenizers)
                )
                router_args.n_tokenizers = len(self.tokenizers)
            router_args.tokenizer_names = tokenizer_names

        # Fix the random init to the tokenizer seed so all dataloader workers
        # (each of which constructs its own RoutedSuperTokenizer) agree on the
        # same initial weights before any checkpoint is loaded.
        seed = kwargs.get("rng_state") if isinstance(kwargs.get("rng_state"), int) else None
        if seed is None:
            seed = 42  # matches SupersetTokenizer's default rng seed
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.router = ByteRouter(router_args)

        if router_checkpoint is not None:
            state = torch.load(router_checkpoint, map_location="cpu")
            self.router.load_state_dict(state)
            logger.info("Loaded ByteRouter weights from %s", router_checkpoint)

    # ------------------------------------------------------------------
    # routing helpers
    # ------------------------------------------------------------------

    def _router_device(self):
        import torch
        try:
            return next(self.router.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def route(self, text: str) -> int:
        """Return the tokenizer index chosen by the learned router."""
        return self.router.route_text(text)

    def route_batch(self, texts: list) -> list:
        """Return tokenizer indices for a list of texts."""
        return self.router.route_texts(texts)

    # ------------------------------------------------------------------
    # override encode to use router when the router has been initialised
    # ------------------------------------------------------------------

    def encode(
        self,
        tokens,
        add_bos: bool,
        add_eos: bool,
        tokenizer_choice=None,
    ):
        if tokenizer_choice is None:
            tokenizer_choice = self.route(tokens)
        return super().encode(tokens, add_bos, add_eos, tokenizer_choice=tokenizer_choice)


def build_token_bytes(tokenizer: "Tokenizer", vocab_size: int) -> Dict[int, int]:
    """Return a dict mapping token_id -> UTF-8 byte length of its surface form.
    Special tokens (bos/eos or ~SPECIAL~ surfaces) are omitted (0 bytes → excluded from BPB).
    """
    special_ids = {getattr(tokenizer, "bos_id", None), getattr(tokenizer, "eos_id", None)}
    special_ids.discard(None)
    result: Dict[int, int] = {}

    if hasattr(tokenizer, "super_vocab"):
        # SupersetTokenizer: super_vocab is surface_string -> token_id
        for surface, token_id in tokenizer.super_vocab.items():
            if token_id >= vocab_size or token_id in special_ids:
                continue
            if surface.startswith("~SPECIAL~"):
                continue
            nb = len(surface.encode("utf-8"))
            if nb > 0:
                result[token_id] = nb
    elif hasattr(tokenizer, "sp_model"):
        sp = tokenizer.sp_model
        for i in range(vocab_size):
            if i in special_ids:
                continue
            piece = sp.id_to_piece(i)
            nb = len(piece.replace("\u2581", " ").encode("utf-8"))
            if nb > 0:
                result[i] = nb
    elif hasattr(tokenizer, "tkt_model"):
        for i in range(vocab_size):
            if i in special_ids:
                continue
            try:
                nb = len(tokenizer.tkt_model.decode_single_token_bytes(i))
                if nb > 0:
                    result[i] = nb
            except Exception:
                pass
    elif hasattr(tokenizer, "hf_tokenizer"):
        for i in range(vocab_size):
            if i in special_ids:
                continue
            try:
                nb = len(tokenizer.decode([i], skip_special_tokens=False).encode("utf-8"))
                if nb > 0:
                    result[i] = nb
            except Exception:
                pass
    else:
        # ByteTokenizer: tokens 0-255 are exactly one byte each
        for i in range(min(256, vocab_size)):
            if i not in special_ids:
                result[i] = 1

    return result


def build_tokenizer(name: str, path: Optional[Union[str, List[Dict[str, str]]]] = None, tokenizers: Optional[List[Dict[str, str]]]=None, dropout: float = 0, rng_state: Dict[str, Any] = None, superset_code_name: Optional[str] = None, n_words: Optional[int] = None, router_args: Optional[Any] = None, router_checkpoint: Optional[str] = None) -> Tokenizer:
    if name == "bytes":
        return ByteTokenizer()
    elif name == "mock":
        return MockTokenizer()
    elif name == "sp":
        return SentencePieceTokenizer(path, alpha=dropout)
    elif name == "tiktoken":
        return TikTokenTokenizer(path)
    elif name == "huggingface" and "byt5" in path:
        return ByT5HFTokenizer(path)
    elif name == "huggingface":
        return SimplifiedHFTokenizer(path, dropout=dropout)
    elif name == "tokenmonster":
        return TokenMonsterTokenizer(path)
    elif name == "tekken":
        return TekkenTokenizer()
    elif name == "supertokenizer":
        return SupersetTokenizer(tokenizers, rng_state=rng_state, superset_code_name=superset_code_name, n_words=n_words)
    elif name == "routed_supertokenizer":
        from lingua.router import ByteRouterArgs
        if router_args is not None and not isinstance(router_args, ByteRouterArgs):
            router_args = ByteRouterArgs(**router_args)
        return RoutedSuperTokenizer(
            tokenizers,
            router_args=router_args,
            router_checkpoint=router_checkpoint,
            rng_state=rng_state,
            superset_code_name=superset_code_name,
            n_words=n_words,
        )
    else:
        raise NotImplementedError(f"{name} tokenizer type is not implemented")
