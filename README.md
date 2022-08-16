# Hilbert-Transformer

## Repository Structure

The repository has the following structure:

```
hilbert_transformer
├── data
│   └── ...
├── hilbert_transformer
│   ├── __init__.py
│   ├── hilbert_embedding.py
│   ├── hilbert_linear.py
│   ├── hilbert_multihead_attention.py
│   ├── hilbert_positional_encoding.py
│   └── hilbert_transformer.py
├── LICENSE
├── notebooks
│   └── ...
├── README.md
└── requirements.txt
```

- `data/` contains the training and evaluation data for the transformer model.
- `hilbert_transformer/` contains the source code for the transformer model.
- `notebooks/` contains Jupyter notebooks that demonstrate how to use the transformer model.
- `requirements.txt` contains the Python package requirements.

## Dependencies

The transformer model requires the following Python packages:

- `numpy`
- `torch`
- `tqdm`
- `transformers`


## Usage

### Training a Transformer Model with a Hilbert Matrix Weight Matrix

To train a transformer model with a Hilbert matrix weight matrix, run the following command:

```bash
python -m hilbert_transformer.hilbert_transformer \
    --order 3 \
    --embedding_dim 2 \
    --hidden_dim 4 \
    --num_heads 2 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --max_source_length 10 \
    --max_target_length 10 \
    --num_epochs 20 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --save_directory checkpoints/hilbert/transformer/order=3/embedding=2/hidden=4/heads=2/encoder=6/decoder=6/max-source=10/max-target=10/epochs=20/batch=32/lr=1e-4/exp-id=`date +%Y%m%d%H%M%S`
```

### Evaluating a Transformer Model with a Hilbert Matrix Weight Matrix

To evaluate a transformer model with a Hilbert matrix weight matrix, run the following command:

```bash
python -m hilbert_transformer.hilbert_transformer \
    --order 3 \
    --embedding_dim 2 \
    --hidden_dim 4 \
    --num_heads 2 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --max_source_length 10 \
    --max_target_length 10 \
    --batch_size 32 \
    --load_directory checkpoints/hilbert/transformer/order=3/embedding=2/hidden=4/heads=2/encoder=6/decoder=6/max-source=10/max-target=10/epochs=20/batch=32/lr=1e-4/exp-id=20200714094818
```

### Visualizing the Attention Weights of a Transformer Model with a Hilbert Matrix Weight Matrix

To visualize the attention weights of a transformer model with a Hilbert matrix weight matrix, run the following command:

```bash
python -m hilbert_transformer.hilbert_transformer \
    --order 3 \
    --embedding_dim 2 \
    --hidden_dim 4 \
    --num_heads 2 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --max_source_length 10 \
    --max_target_length 10 \
    --batch_size 32 \
    --load_directory checkpoints/hilbert/transformer/order=3/embedding=2/hidden=4/heads=2/encoder=6/decoder=6/max-source=10/max-target=10/epochs=20/batch=32/lr=1e-4/exp-id=20200714094818 \
    --visualize 1
```

### Computing the Sensitivity of a Transformer Model with a Hilbert Matrix Weight Matrix

To compute the sensitivity of a transformer model with a Hilbert matrix weight matrix, run the following command:

```bash
python -m hilbert_transformer.hilbert_transformer \
    --order 3 \
    --embedding_dim 2 \
    --hidden_dim 4 \
    --num_heads 2 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --max_source_length 10 \
    --max_target_length 10 \
    --batch_size 32 \
    --load_directory checkpoints/hilbert/transformer/order=3/embedding=2/hidden=4/heads=2/encoder=6/decoder=6/max-source=10/max-target=10/epochs=20/batch=32/lr=1e-4/exp-id=20200714094818 \
    --compute_sensitivity 1
