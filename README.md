# Transformer from scratch


The model is trained to translate English to Italian using the opus_books dataset from Hugging Face.

Paper: https://arxiv.org/abs/1706.03762

## What's in here

- `model.py` - the whole architecture: input embeddings, positional encoding, layer norm, feed forward block, multi-head attention, residual connections, encoder, decoder, projection layer, and `build_transformer()` which puts it all together
- `dataset.py` - dataset class that tokenizes sentence pairs, adds the start/end/pad tokens and builds the masks
- `train.py` - tokenizer training, dataloaders, training loop, greedy decoding and validation
- `config.py` - hyperparameters and helpers for checkpoint paths
- `tokenizers/` - word level tokenizers for English and Italian
- `Transformer_from_scratch.ipynb` - notebook I used while working on it

## Quick overview of how it works

The encoder takes the English sentence and runs it through self-attention + feed forward layers (3 layers here) so every word gets context from the rest of the sentence.

The decoder generates the Italian sentence one token at a time. Each layer does masked self-attention over the tokens generated so far (the mask stops it from looking at future tokens), then cross-attention over the encoder output, then a feed forward layer.

At the end a linear layer turns the decoder output into probabilities over the Italian vocabulary and the most likely token is picked.

During training I use teacher forcing, so the decoder gets the correct previous tokens as input and has to predict the next one.

## Running it

Install the dependencies:

```
pip install torch datasets tokenizers tensorboard tqdm
```

Then start training:

```
python train.py
```

On the first run it downloads the dataset and builds the tokenizers if they aren't there already. A checkpoint is saved after every epoch in `opus_books_weights/`, and a couple of validation sentences are printed so you can see how the translations improve.

To continue from a saved checkpoint, set `preload` in `config.py` to `"latest"` or to an epoch number like `"05"`.

To look at the loss curve:

```
tensorboard --logdir runs
```

You really want a GPU for this. It works on CPU but it's very slow.

## Settings

Defaults I used (in `config.py` and `build_transformer()`):

- d_model = 512
- 8 attention heads
- 3 encoder and 3 decoder layers
- feed forward size = 2048
- dropout = 0.1
- seq_len = 350
- batch size = 4
- learning rate = 1e-4 with Adam
- 20 epochs

## Differences from the paper

- I used pre-norm residual connections (normalize before the sublayer) instead of post-norm, since it trains more stably
- 3 layers instead of 6 to keep training time reasonable
- Label smoothing of 0.1 in the loss, and padding tokens are ignored
- Greedy decoding instead of beam search

