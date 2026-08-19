# Frozen model architectures

These are the exact architectures of the replacement experiments. They are not asserted to be
the deleted historical implementations. The three compact models were frozen before inspecting
new test predictions, and their trainable parameter counts reproduce the counts preserved in the
older manuscript.

## Common input and output

- Compact models receive a batch of onset-aligned log-magnitude STFT tensors with shape
  `B × 1 × 257 × T`. The time dimension varies with prefix duration and is right-padded only
  within a batch; masks or exact lengths prevent padded positions from contributing.
- Every classifier emits two unnormalized logits ordered as control and dysarthric.
- Convolutions use no bias when followed by batch normalization.
- Unless stated otherwise, dropout is 0.30.

## Small CNN — 23,538 trainable parameters

| Stage | Exact operation | Output channels/features | Parameters |
|---|---|---:|---:|
| 1 | Conv2d `1→16`, `3×3`, stride 1, padding 1; BatchNorm2d; SiLU; MaxPool `2×2` | 16 | 176 |
| 2 | Conv2d `16→32`, `3×3`, stride 1, padding 1; BatchNorm2d; SiLU; MaxPool `2×2` | 32 | 4,672 |
| 3 | Conv2d `32→64`, `3×3`, stride 1, padding 1; BatchNorm2d; SiLU; MaxPool `2×2` | 64 | 18,560 |
| Pool | Length-aware mean over all valid frequency/time positions | 64 | 0 |
| Head | Dropout 0.30; Linear `64→2` | 2 | 130 |

The three pools reduce both frequency and time by a factor of two at each stage. For mixed-length
batches, padded STFT values are zeroed before convolution and only valid downsampled positions
enter the final mean.

## CNN–BiLSTM — 69,354 trainable parameters

| Stage | Exact operation | Output shape/features | Parameters |
|---|---|---:|---:|
| 1 | Conv2d `1→24`, `3×3`, stride 1, padding 1; BatchNorm2d; SiLU; MaxPool `2×1` | `B×24×128×T` | 264 |
| 2 | Conv2d `24→48`, `3×3`, stride 1, padding 1; BatchNorm2d; SiLU; MaxPool `2×1` | `B×48×64×T` | 10,464 |
| Tokens | Mean over the remaining 64 frequency bins, then transpose | `B×T×48` | 0 |
| Sequence | One-layer bidirectional LSTM, input 48, hidden 64 per direction | `B×T×128` | 58,368 |
| Head | Masked temporal mean; Dropout 0.30; Linear `128→2` | 2 | 258 |

The convolutional pools operate only on frequency, so one recurrent token is retained for every
input STFT frame. Mixed-length batches use packed sequences and masked pooling.

## Short-prefix Transformer — 196,594 trainable parameters

| Stage | Exact operation | Output shape/features | Parameters |
|---|---|---:|---:|
| 1 | Conv2d `1→24`, `3×3`, stride 1, padding 1; BatchNorm2d; SiLU; MaxPool `2×1` | `B×24×128×T` | 264 |
| 2 | Conv2d `24→48`, `3×3`, stride 1, padding 1; BatchNorm2d; SiLU; MaxPool `2×1` | `B×48×64×T` | 10,464 |
| Tokens | Adaptive mean-pool frequency to 2 bins; flatten `48×2`; bias-free Linear `96→96`; add sinusoidal position encoding | `B×T×96` | 9,216 |
| Encoder | Two Transformer encoder layers, model dimension 96, 4 attention heads, feed-forward dimension 256, GELU, dropout 0.30, post-norm | `B×T×96` | 174,272 |
| Head | Masked temporal mean; Dropout; Linear `96→24`; SiLU; Dropout; Linear `24→2` | 2 | 2,378 |

The tokenizer never pools time. The Transformer receives a `src_key_padding_mask`, padded encoded
states are cleared, and only valid tokens enter the final mean.

## WavLM Base+ modern reference

- Checkpoint: `microsoft/wavlm-base-plus`, pinned revision
  `4c66d4806a428f2e922ccfa1a962776e232d487b`. The loader also records the resolved commit in
  each run summary and requires it to match the requested identity.
- Input: raw mono 16 kHz onset-aligned waveform with an exact 1/0 sample attention mask.
- Representation: final WavLM hidden states.
- Pooling: mean over valid downsampled feature positions only.
- Primary feasible mode: freeze the entire pretrained backbone; train Dropout 0.30 plus Linear
  `hidden_size→2`. With the standard 768-dimensional Base+ representation, the classification
  layer has 1,538 trainable parameters. The loaded run records both total and trainable counts.
- Optional sensitivity mode: unfreeze only the final `N` WavLM encoder layers. Any such result
  must be labelled separately from the frozen-backbone reference.

## Removed historical model names

The submitted manuscript named CNN-GRU and “EfficientNet-lite,” but no layer definitions,
variant identifier, checkpoints, or prediction files survived. Those results are removed from
the reproducible primary comparison. Assigning a convenient EfficientNet-Lite variant after the
fact would not recover the deleted experiment.
