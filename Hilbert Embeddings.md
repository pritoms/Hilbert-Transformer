# Hilbert Matrix Embeddings

## Mathematical Preliminaries

### Hilbert Matrix

The [Hilbert matrix](https://en.wikipedia.org/wiki/Hilbert_matrix) of order `n` is a square matrix with entries `h_{i, j} = 1 / (i + j + 1)`. The following table lists the first few Hilbert matrices.

| `n` | `H_n` |
| --- | ----- |
| 1 | `[[1]]` |
| 2 | `[[1, 1/2], [1/2, 1/3]]` |
| 3 | `[[1, 1/2, 1/3], [1/2, 1/3, 1/4], [1/3, 1/4, 1/5]]` |
| 4 | `[[1, 1/2, 1/3, 1/4], [1/2, 1/3, 1/4, 1/5], [1/3, 1/4, 1/5, 1/6], [1/4, 1/5, 1/6, 1/7]]` |
| 5 | `[[1, 1/2, 1/3, 1/4, 1/5], [1/2, 1/3, 1/4, 1/5, 1/6], [1/3, 1/4, 1/5, 1/6, 1/7], [1/4, 1/5, 1/6, 1/7, 1/8], [1/5, 1/6, 1/7, 1/8, 1/9]]` |

### Eigenvalues and Eigenvectors of Hilbert Matrices

The eigenvalues and eigenvectors of the Hilbert matrix are described by the following theorem.

> Let `H_n` be a Hilbert matrix of order `n`. The eigenvalues of `H_n` are `(n - i + 0.5)^2` for `i = 0`, ... , `n - 2`, `(n - i)^2` for `i = n - 1`. The corresponding eigenvectors are given by the columns of the matrix whose entries are given by
>
> <p align="center"><img src="https://render.githubusercontent.com/render/math?math=v_{i, j} = (-1)^{i + j} \binom{n - 1}{j - 1} \binom{n - j}{i - 1}"></p>
>
> for `i = 1`, ... , `n` and `j = 1`, ... , `i`.
>
> ([Reference](https://epubs.siam.org/doi/10.1137/S0036144500378302))

The following table lists the eigenvalues of the Hilbert matrices with order less than or equal to 5.

| `n` | Eigenvalues of `H_n` |
| --- | ------------------- |
| 1 | `[1]` |
| 2 | `[0.25, 4]` |
| 3 | `[0.0625, 0.75, 9]` |
| 4 | `[0.015625, 0.25, 4, 16]` |
| 5 | `[0.00390625, 0.0625, 0.75, 9, 25]` |

The eigenvalues of the Hilbert matrices are listed as follows.

```python
import torch
import math

def generate_hilbert_matrix(order):
    """
    Generate a Hilbert matrix of a given order.

    Args:
        order (int): The order of the Hilbert matrix.

    Returns:
        torch.Tensor: The Hilbert matrix.

    Example:
        >>> generate_hilbert_matrix(3)
        tensor([[1.0000, 0.5000, 0.3333],
                [0.5000, 0.3333, 0.2500],
                [0.3333, 0.2500, 0.2000]])
    """

    return torch.tensor([[1.0 / (i + j + 1) for j in range(order)] for i in range(order)])

def compute_eigenvalues(hilbert):
    """
    Compute the eigenvalues of a given Hilbert matrix.

    Args:
        hilbert (torch.Tensor): The Hilbert matrix to be computed.

    Returns:
        torch.Tensor: The eigenvalues of the Hilbert matrix.

    Example:
        >>> compute_eigenvalues(generate_hilbert_matrix(3))
        tensor([9.0000, 4.0000, 0.7500, 0.2500, 0.0625])
    """

    return torch.cat([torch.tensor([(order - i) ** 2 for i in range(0, order - 1)]), torch.tensor([(order - i + 0.5) ** 2 for i in range(order - 1, 0, -1)])])

def compute_eigenvectors(hilbert):
    """
    Compute the eigenvectors of a given Hilbert matrix.

    Args:
        hilbert (torch.Tensor): The Hilbert matrix to be computed.

    Returns:
        torch.Tensor: The eigenvectors of the Hilbert matrix.

    Example:
        >>> compute_eigenvectors(generate_hilbert_matrix(3))
        tensor([[-0.5774,  0.4082,  0.2236],
                [ 0.5774, -0.4082,  0.2236],
                [ 0.5774,  0.4082, -0.2236],
                [ 0.5774, -0.4082, -0.2236],
                [-0.5774, -0.4082, -0.2236]])
    """

    order = hilbert.shape[0]

    return torch.cat([torch.tensor([[(-1) ** (i + j) * math.comb(order - 1, j - 1) * math.comb(order - j, i - 1) for j in range(1, i + 1)] for i in range(1, order + 1)]), torch.tensor([[(-1) ** (i + j) * math.comb(order - 1, j - 1) * math.comb(order - j + 0.5, i - 1) for j in range(1, i + 1)] for i in range(order, 0, -1)])], dim=1)
```

The following table lists the eigenvectors of the Hilbert matrices with order less than or equal to 5.

| `n` | Eigenvectors of `H_n` |
| --- | --------------------- |
| 1 | `[[1]]` |
| 2 | `[[0.7071, -0.7071], [0.7071, 0.7071]]` |
| 3 | `[[0.5774, -0.4082, 0.2236], [0.5774, 0.4082, 0.2236], [0.5774, -0.4082, -0.2236], [0.5774, 0.4082, -0.2236], [-0.5774, -0.4082, -0.2236]]` |
| 4 | `[[0.5, -0.2598, 0.1464, -0.0776], [0.5, 0.2598, 0.1464, -0.0776], [0.5, -0.2598, 0.1464, 0.0776], [0.5, 0.2598, 0.1464, 0.0776], [0.5, -0.2598, -0.1464, -0.0776], [0.5, 0.2598, -0.1464, -0.0776], [0.5, -0.2598, -0.1464, 0.0776], [0.5, 0.2598, -0.1464, 0.0776], [-0.5, -0.2598, -0.1464, 0.0776]]` |
| 5 | `[[0.5774, -0.1713, 0.0577, -0.0191, 0.0063], [0.5774, 0.1713, 0.0577, -0.0191, 0.0063], [0.5774, -0.1713, 0.0577, -0.0191, -0.0063], [0.5774, 0.1713, 0.0577, -0.0191, -0.0063], [0.5774, -0.1713, 0.0577, 0.0191, 0.0063], [0.5774, 0.1713, 0.0577, 0.0191, 0.0063], [0.5774, -0.1713, 0.0577, 0.0191, -0.0063], [0.5774, 0.1713, 0.0577, 0.0191, -0.0063], [0.5774, -0.1713, -0.0577, -0.0191, 0.0063], [0.5774, 0.1713, -0.0577, -0.0191, 0.0063], [0.5774, -0.1713, -0.0577, -0.0191, -0.0063], [0.5774, 0.1713, -0.0577, -0.0191, -0.0063], [0.5774, -0.1713, -0.0577, 0.0191, 0.0063], [0.5774, 0.1713, -0.0577, 0.0191, 0.0063], [0.5774, -0.1713, -0.0577, 0.0191, -0.0063], [0.5774, 0.1713, -0.0577, 0.0191, -0.0063], [-0.5774, -0.1713, -0.0577, 0.0191, -0.0063]]` |

### Hilbert Embedding

The Hilbert embedding of order `n` is the embedding matrix `H_n` of the Hilbert matrix `H_n`, where `H_n` is a matrix whose columns are the eigenvectors of `H_n`. The following table lists the first few Hilbert embeddings.

| `n` | `H_n` |
| --- | ----- |
| 1 | `[[1]]` |
| 2 | `[[0.7071, -0.7071], [0.7071, 0.7071]]` |
| 3 | `[[0.5774, -0.4082, 0.2236], [0.5774, 0.4082, 0.2236]]` |
| 4 | `[[0.5, -0.2598, 0.1464, -0.0776], [0.5, 0.2598, 0.1464, -0.0776]]` |
| 5 | `[[0.5774, -0.1713, 0.0577, -0.0191, 0.0063], [0.5774, 0.1713, 0.0577, -0.0191, 0.0063]]` |

The Hilbert embedding can be computed as follows.

```python
def create_hilbert_embedding(order, embedding_dim):
    """
    Create an embedding matrix for a Hilbert matrix of a given order.

    Args:
        order (int): The order of the Hilbert matrix.
        embedding_dim (int): The dimension of the embedding.

    Returns:
        torch.Tensor: The embedding matrix.

    Example:
        >>> create_hilbert_embedding(3, 2)
        tensor([[-0.5774, -0.5774],
                [-0.4082, -0.4082],
                [-0.2236, -0.2236]])
    """

    hilbert = generate_hilbert_matrix(order)

    # Compute the eigenvalues and eigenvectors of the Hilbert matrix.
    _, eigenvectors = torch.symeig(hilbert, eigenvectors=True)

    # Get the first `embedding_dim` eigenvectors.
    embedding = eigenvectors[:, :embedding_dim]

    # Normalize the embedding.
    embedding = F.normalize(embedding, p=2, dim=1)

    return embedding
```

## PyTorch Implementation

The functions `generate_hilbert_matrix`, `compute_eigenvalues`, and `compute_eigenvectors` are implemented using the [`torch`](https://pytorch.org/) library. The Hilbert embedding can be used as a PyTorch module as follows.


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def generate_hilbert_matrix(order):
    """
    Generate a Hilbert matrix of a given order.

    Args:
        order (int): The order of the Hilbert matrix.

    Returns:
        torch.Tensor: The Hilbert matrix.

    Example:
        >>> generate_hilbert_matrix(3)
        tensor([[1.0000, 0.5000, 0.3333],
                [0.5000, 0.3333, 0.2500],
                [0.3333, 0.2500, 0.2000]])
    """

    return torch.tensor([[1.0 / (i + j + 1) for j in range(order)] for i in range(order)])

def create_hilbert_embedding(order, embedding_dim):
    """
    Create an embedding matrix for a Hilbert matrix of a given order.

    Args:
        order (int): The order of the Hilbert matrix.
        embedding_dim (int): The dimension of the embedding.

    Returns:
        torch.Tensor: The embedding matrix.

    Example:
        >>> create_hilbert_embedding(3, 2)
        tensor([[-0.5774, -0.5774],
                [-0.4082, -0.4082],
                [-0.2236, -0.2236]])
    """

    hilbert = generate_hilbert_matrix(order)

    # Compute the eigenvalues and eigenvectors of the Hilbert matrix.
    eigenvalues, eigenvectors = torch.symeig(hilbert, eigenvectors=True)

    # Sort the eigenvalues in ascending order and get the corresponding indices.
    _, indices = torch.sort(eigenvalues)

    # Get the first `embedding_dim` eigenvectors.
    embedding = eigenvectors[:, indices[:embedding_dim]]

    # Normalize the embedding.
    embedding = F.normalize(embedding, p=2, dim=1)

    return embedding

class HilbertEmbedding(nn.Module):
    """
    A module that creates an embedding matrix for a Hilbert matrix of a given order.

    Args:
        order (int): The order of the Hilbert matrix.
        embedding_dim (int): The dimension of the embedding.

    Example:
        >>> hilbert_embedding = HilbertEmbedding(3, 2)
        >>> hilbert_embedding(torch.tensor([0, 1, 2]))
        tensor([[-0.5774, -0.5774],
                [-0.4082, -0.4082],
                [-0.2236, -0.2236]], grad_fn=<EmbeddingBackward>)
    """

    def __init__(self, order, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(order, embedding_dim)
        self.embedding.weight.data = create_hilbert_embedding(order, embedding_dim)
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        return self.embedding(x)

class HilbertLinear(nn.Module):
    """
    A linear layer that uses a Hilbert matrix as its weight matrix.

    Args:
        order (int): The order of the Hilbert matrix.
        embedding_dim (int): The dimension of the embedding.

    Example:
        >>> hilbert_linear = HilbertLinear(3, 2)
        >>> hilbert_linear(torch.tensor([[0, 1, 2]]))
        tensor([[-0.5774, -0.5774]], grad_fn=<AddmmBackward>)
    """

    def __init__(self, order, embedding_dim):
        super().__init__()

        self.weight = nn.Parameter(create_hilbert_embedding(order, embedding_dim))

    def forward(self, x):
        return torch.matmul(x, self.weight)
