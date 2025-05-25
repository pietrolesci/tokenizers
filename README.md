
## Modifications Explained

This fork of the `tokenizers` library patches the `src/models/bpe/trainer.rs` (used to train BPE and WordPiece tokenisers) file such that it logs the merge pairs and their counts to files in the current working directory.
Specifically, it creates:

- `all_merges.jsonl`: despite the name (I thought these where "all" merges), these are the initial pairs extracted from words (note: words are computed by applying pretokenisation, e.g., whitespace splitting). This file is *not* what you want, yet I left it there just in case.

- `implemented_merges.jsonl`: this is the file containing all merge pairs and their counts. This is the file that you want.


## Quickstart and Example
To get started, simply clone this repo and pip install it. Then, you can train a tokeniser from python. 

For example, you can get a BPE or WordPiece tokeniser as follows

```python
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

MAX_VOCAB_SIZE = 32_000
EOS_TOKEN = "<|endoftext|>"
UNK_TOKEN = "<|unk|>"

def get_tok(tok_type: str) -> tuple[Tokenizer, trainers.BpeTrainer | trainers.WordPieceTrainer]:
    # Define the tokenizer and set up the tokenizer components (this is a GPT-2 tokenizer)
    # https://github.com/huggingface/tokenizers/blob/14a07b06e4a8bd8f80d884419ae4630f5a3d8098/bindings/python/py_src/tokenizers/implementations/byte_level_bpe.py#L10
    assert tok_type in ["bpe", "wordpiece"]

    model_cls = models.BPE if tok_type == "bpe" else partial(models.WordPiece, unk_token=UNK_TOKEN)
    trainer_cls = trainers.BpeTrainer if tok_type == "bpe" else trainers.WordPieceTrainer

    # Tokenizer
    tokenizer = Tokenizer(model_cls())  # type: ignore
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)  # type: ignore
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)  # type: ignore
    tokenizer.decoder = decoders.ByteLevel()  # type: ignore

    # Trainer
    kwargs = {
        "vocab_size": MAX_VOCAB_SIZE,
        "min_frequency": 2,
        "special_tokens": [EOS_TOKEN] if tok_type == "bpe" else [EOS_TOKEN, UNK_TOKEN],
        "show_progress": True,
        "initial_alphabet": pre_tokenizers.ByteLevel.alphabet(),
    }
    trainer = trainer_cls(**kwargs)

    return tokenizer, trainer
```

And you can train it on a dataset of documents as follows

```python
import srsly
from tokenizers import Tokenizer


# Get tokenizer model and its trainer
tokenizer, trainer = get_tok("bpe")

# Train on dataset: dict[str, str]. Pass in the len of documents if you want a progress bar
tokenizer.train_from_iterator(iter(x["text"] for x in dataset), trainer, len(dataset))

# Save it to disk
tokenizer.save("tokenizer.json", pretty=True)

# Since the special characters are not saved by default, save them manually. I am not sure
# whether this is now fixed
meta = {"max_vocab_size": MAX_VOCAB_SIZE, "eos_token": EOS_TOKEN, "unk_token": UNK_TOKEN}
srsly.write_yaml(folder_path / "metadata.yaml", meta)
```

Finally you can load the tokenizer in the `transformers` library as follows

```python
import srsly
import json
from transformers import PreTrainedTokenizerFast 


# Read the conf, the metadata that are needed, and instantiate tokenizer object
conf: dict = srsly.read_json(path / "tokenizer.json")
backend_tok = Tokenizer.from_str(json.dumps(conf))
meta = srsly.read_yaml(path / "metadata.yaml")
eos_token: str = meta["eos_token"]  # type: ignore
unk_token: str | None = meta.get("unk_token", None)  # type: ignore
   
# Instantiate PreTrainedTokenizerFast (from `transformers` library) from tokenizer object
# NOTE: we do not instantiate from file directly due to compatibility
# https://github.com/huggingface/tokenizers/issues/1562#issuecomment-2315349846
tok = PreTrainedTokenizerFast(tokenizer_object=backend_tok, clean_up_tokenization_spaces=True)

# Add common configs for decoder-only models
tok.padding_side = "left"
tok.eos_token = eos_token
tok.unk_token = unk_token

# And finally you can save it as a `transformers`'s tokenizer
tok.save_pretrained(...)
```

----

<p align="center">
    <br>
    <img src="https://huggingface.co/landing/assets/tokenizers/tokenizers-logo.png" width="600"/>
    <br>
<p>
<p align="center">
    <img alt="Build" src="https://github.com/huggingface/tokenizers/workflows/Rust/badge.svg">
    <a href="https://github.com/huggingface/tokenizers/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/tokenizers.svg?color=blue&cachedrop">
    </a>
    <a href="https://pepy.tech/project/tokenizers">
        <img src="https://pepy.tech/badge/tokenizers/week" />
    </a>
</p>

Provides an implementation of today's most used tokenizers, with a focus on performance and
versatility.

## Main features:

 - Train new vocabularies and tokenize, using today's most used tokenizers.
 - Extremely fast (both training and tokenization), thanks to the Rust implementation. Takes
   less than 20 seconds to tokenize a GB of text on a server's CPU.
 - Easy to use, but also extremely versatile.
 - Designed for research and production.
 - Normalization comes with alignments tracking. It's always possible to get the part of the
   original sentence that corresponds to a given token.
 - Does all the pre-processing: Truncate, Pad, add the special tokens your model needs.

## Performances
Performances can vary depending on hardware, but running the [~/bindings/python/benches/test_tiktoken.py](bindings/python/benches/test_tiktoken.py) should give the following on a g6 aws instance:
![image](https://github.com/user-attachments/assets/2b913d4b-e488-4cbc-b542-f90a6c40643d)


## Bindings

We provide bindings to the following languages (more to come!):
  - [Rust](https://github.com/huggingface/tokenizers/tree/main/tokenizers) (Original implementation)
  - [Python](https://github.com/huggingface/tokenizers/tree/main/bindings/python)
  - [Node.js](https://github.com/huggingface/tokenizers/tree/main/bindings/node)
  - [Ruby](https://github.com/ankane/tokenizers-ruby) (Contributed by @ankane, external repo)
 
## Quick example using Python:

Choose your model between Byte-Pair Encoding, WordPiece or Unigram and instantiate a tokenizer:

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE())
```

You can customize how pre-tokenization (e.g., splitting into words) is done:

```python
from tokenizers.pre_tokenizers import Whitespace

tokenizer.pre_tokenizer = Whitespace()
```

Then training your tokenizer on a set of files just takes two lines of codes:

```python
from tokenizers.trainers import BpeTrainer

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(files=["wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"], trainer=trainer)
```

Once your tokenizer is trained, encode any text with just one line:
```python
output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)
# ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"]
```

Check the [documentation](https://huggingface.co/docs/tokenizers/index)
or the [quicktour](https://huggingface.co/docs/tokenizers/quicktour) to learn more!
