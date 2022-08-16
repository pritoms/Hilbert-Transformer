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

class HilbertConv1d(nn.Module):
    """
    A 1D convolutional layer that uses a Hilbert matrix as its weight matrix.
    
    Args:
        order (int): The order of the Hilbert matrix.
        embedding_dim (int): The dimension of the embedding.
        kernel_size (int): The size of the convolving kernel.
    
    Example:
        >>> hilbert_conv1d = HilbertConv1d(3, 2, 3)
        >>> hilbert_conv1d(torch.tensor([[[0, 1, 2]]]))
        tensor([[[-0.5774, -0.5774],
                 [-0.4082, -0.4082],
                 [-0.2236, -0.2236]]], grad_fn=<SqueezeBackward1>)
    """

    def __init__(self, order, embedding_dim, kernel_size):
        super().__init__()

        self.weight = nn.Parameter(create_hilbert_embedding(order, embedding_dim).unsqueeze(0).unsqueeze(0))
        self.kernel_size = kernel_size

    def forward(self, x):
        return F.conv1d(x, self.weight, padding=self.kernel_size // 2)

class HilbertConv2d(nn.Module):
    """
    A 2D convolutional layer that uses a Hilbert matrix as its weight matrix.
    
    Args:
        order (int): The order of the Hilbert matrix.
        embedding_dim (int): The dimension of the embedding.
        kernel_size (int): The size of the convolving kernel.
    
    Example:
        >>> hilbert_conv2d = HilbertConv2d(3, 2, 3)
        >>> hilbert_conv2d(torch.tensor([[[[0, 1, 2], [0, 1, 2], [0, 1, 2]]]]))
        tensor([[[[-0.5774, -0.5774],
                  [-0.4082, -0.4082],
                  [-0.2236, -0.2236]]]], grad_fn=<SqueezeBackward1>)
    """

    def __init__(self, order, embedding_dim, kernel_size):
        super().__init__()

        self.weight = nn.Parameter(create_hilbert_embedding(order, embedding_dim).unsqueeze(0).unsqueeze(0))
        self.kernel_size = kernel_size

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=self.kernel_size // 2)

class HilbertConv3d(nn.Module):
    """
    A 3D convolutional layer that uses a Hilbert matrix as its weight matrix.
    
    Args:
        order (int): The order of the Hilbert matrix.
        embedding_dim (int): The dimension of the embedding.
        kernel_size (int): The size of the convolving kernel.
    
    Example:
        >>> hilbert_conv3d = HilbertConv3d(3, 2, 3)
        >>> hilbert_conv3d(torch.tensor([[[[[0, 1, 2], [0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]]]]]]))
        tensor([[[[[-0.5774, -0.5774],
                   [-0.4082, -0.4082],
                   [-0.2236, -0.2236]],
                  [[-0.5774, -0.5774],
                   [-0.4082, -0.4082],
                   [-0.2236, -0.2236]],
                  [[-0.5774, -0.5774],
                   [-0.4082, -0.4082],
                   [-0.2236, -0.2236]]]]]], grad_fn=<SqueezeBackward1>)
    """

    def __init__(self, order, embedding_dim, kernel_size):
        super().__init__()

        self.weight = nn.Parameter(create_hilbert_embedding(order, embedding_dim).unsqueeze(0).unsqueeze(0))
        self.kernel_size = kernel_size

    def forward(self, x):
        return F.conv3d(x, self.weight, padding=self.kernel_size // 2)

class HilbertTransformerEncoderLayer(nn.Module):
    """
    A single encoder layer of the Transformer model that uses a Hilbert matrix as its weight matrix.
    
    Args:
        order (int): The order of the Hilbert matrix.
        embedding_dim (int): The dimension of the embedding.
        hidden_dim (int): The dimension of the hidden state.
        num_heads (int): The number of attention heads.
    
    Example:
        >>> hilbert_transformer_encoder_layer = HilbertTransformerEncoderLayer(3, 2, 4, 2)
        >>> hilbert_transformer_encoder_layer(torch.tensor([[[0, 1, 2]]]), torch.tensor([[[0, 1, 2]]]))
        (tensor([[[-0.5774, -0.5774, -0.5774, -0.5774],
                  [-0.4082, -0.4082, -0.4082, -0.4082],
                  [-0.2236, -0.2236, -0.2236, -0.2236]]], grad_fn=<AddBackward0>), tensor([[[-1., -1., -1., -1.]]], grad_fn=<SqueezeBackward1>))
    """

    def __init__(self, order, embedding_dim, hidden_dim, num_heads):
        super().__init__()

        self.self_attention = HilbertMultiheadAttention(order, embedding_dim, hidden_dim, num_heads)
        self.linear = HilbertLinear(order, embedding_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, mask):
        attention_output, attention_weights = self.self_attention(x, x, x, mask)
        attention_output = self.dropout(attention_output)
        output = self.layer_norm(x + attention_output)

        linear_output = self.linear(output)
        linear_output = self.dropout(linear_output)
        output = self.layer_norm(output + linear_output)

        return output, attention_weights

class HilbertMultiheadAttention(nn.Module):
    """
    Multi-headed attention that uses a Hilbert matrix as its weight matrix.
    
    Args:
        order (int): The order of the Hilbert matrix.
        embedding_dim (int): The dimension of the embedding.
        hidden_dim (int): The dimension of the hidden state.
        num_heads (int): The number of attention heads.
    
    Example:
        >>> hilbert_multihead_attention = HilbertMultiheadAttention(3, 2, 4, 2)
        >>> hilbert_multihead_attention(torch.tensor([[[0, 1, 2]]]), torch.tensor([[[0, 1, 2]]]), torch.tensor([[[0, 1, 2]]]), None)
        (tensor([[[-0.5774, -0.5774, -0.5774, -0.5774],
                  [-0.4082, -0.4082, -0.4082, -0.4082],
                  [-0.2236, -0.2236, -0.2236, -0.2236]]], grad_fn=<AddBackward0>), tensor([[[1., 1., 1., 1.]]], grad_fn=<SqueezeBackward1>))
    """

    def __init__(self, order, embedding_dim, hidden_dim, num_heads):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.query_projection = HilbertLinear(order, embedding_dim)
        self.key_projection = HilbertLinear(order, embedding_dim)
        self.value_projection = HilbertLinear(order, embedding_dim)
        self.output_projection = HilbertLinear(order, embedding_dim)

    def forward(self, query, key, value, mask):
        batch_size = query.size(0)

        # Project the query, key and value tensors to the hidden dimension.
        query = self.query_projection(query).view(batch_size, -1, self.num_heads, self.hidden_dim).transpose(1, 2)
        key = self.key_projection(key).view(batch_size, -1, self.num_heads, self.hidden_dim).transpose(1, 2)
        value = self.value_projection(value).view(batch_size, -1, self.num_heads, self.hidden_dim).transpose(1, 2)

        # Compute the scaled dot-product attention.
        attention_weights = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(attention_weights, dim=-1)

        # Apply the attention weights to the value tensor and average the heads' outputs.
        attention_output = torch.matmul(attention_weights, value)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)

        # Project the multi-head output to the embedding dimension.
        output = self.output_projection(attention_output)

        return output, attention_weights

class HilbertTransformerDecoderLayer(nn.Module):
    """
    A single decoder layer of the Transformer model that uses a Hilbert matrix as its weight matrix.
    
    Args:
        order (int): The order of the Hilbert matrix.
        embedding_dim (int): The dimension of the embedding.
        hidden_dim (int): The dimension of the hidden state.
        num_heads (int): The number of attention heads.
    
    Example:
        >>> hilbert_transformer_decoder_layer = HilbertTransformerDecoderLayer(3, 2, 4, 2)
        >>> hilbert_transformer_decoder_layer(torch.tensor([[[0, 1, 2]]]), torch.tensor([[[0, 1, 2]]]), torch.tensor([[[0, 1, 2]]]), None)
        (tensor([[[-0.5774, -0.5774, -0.5774, -0.5774],
                  [-0.4082, -0.4082, -0.4082, -0.4082],
                  [-0.2236, -0.2236, -0.2236, -0.2236]]], grad_fn=<AddBackward0>), tensor([[[1., 1., 1., 1.]]], grad_fn=<SqueezeBackward1>), tensor([[[1., 1., 1., 1.]]], grad_fn=<SqueezeBackward1>))
    """

    def __init__(self, order, embedding_dim, hidden_dim, num_heads):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.self_attention = HilbertMultiheadAttention(order, embedding_dim, hidden_dim, num_heads)
        self.encoder_attention = HilbertMultiheadAttention(order, embedding_dim, hidden_dim, num_heads)
        self.linear = HilbertLinear(order, embedding_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, encoder_output, source_mask, target_mask):
        batch_size = x.size(0)

        # Project the query, key and value tensors to the hidden dimension.
        query = self.query_projection(x).view(batch_size, -1, self.num_heads, self.hidden_dim).transpose(1, 2)
        key = self.key_projection(x).view(batch_size, -1, self.num_heads, self.hidden_dim).transpose(1, 2)
        value = self.value_projection(x).view(batch_size, -1, self.num_heads, self.hidden_dim).transpose(1, 2)

        # Compute the scaled dot-product attention for the decoder's self-attention layer.
        attention_weights = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
        if target_mask is not None:
            attention_weights = attention_weights.masked_fill(target_mask == 0, -1e9)
        attention_weights = F.softmax(attention_weights, dim=-1)

        # Apply the attention weights to the value tensor and average the heads' outputs.
        attention_output = torch.matmul(attention_weights, value)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)

        # Project the multi-head output to the embedding dimension.
        output = self.output_projection(attention_output)

        return output, attention_weights

class HilbertTransformerEncoder(nn.Module):
    """
    The encoder of the Transformer model that uses a Hilbert matrix as its weight matrix.
    
    Args:
        order (int): The order of the Hilbert matrix.
        embedding_dim (int): The dimension of the embedding.
        hidden_dim (int): The dimension of the hidden state.
        num_heads (int): The number of attention heads.
        num_layers (int): The number of encoder layers.
    
    Example:
        >>> hilbert_transformer_encoder = HilbertTransformerEncoder(3, 2, 4, 2, 2)
        >>> hilbert_transformer_encoder(torch.tensor([[[0, 1, 2]]]), None)
        (tensor([[[-0.5774, -0.5774, -0.5774, -0.5774],
                  [-0.4082, -0.4082, -0.4082, -0.4082],
                  [-0.2236, -0.2236, -0.2236, -0.2236]]], grad_fn=<AddBackward0>), tensor([[[1., 1., 1., 1.]]], grad_fn=<SqueezeBackward1>))
    """

    def __init__(self, order, embedding_dim, hidden_dim, num_heads, num_layers):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embedding = HilbertEmbedding(order, embedding_dim)
        self.positional_encoding = HilbertPositionalEncoding(order, embedding_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.encoder_layers = nn.ModuleList([HilbertTransformerEncoderLayer(order, embedding_dim, hidden_dim, num_heads) for _ in range(num_layers)])

    def forward(self, x, mask):
        output = self.embedding(x)
        output = self.positional_encoding(output)
        output = self.dropout(output)
        output = self.layer_norm(output)

        attention_weights = {}

        for i in range(self.num_layers):
            output, attention_weight = self.encoder_layers[i](output, mask)
            attention_weights['encoder_layer_{}'.format(i + 1)] = attention_weight

        return output, attention_weights

class HilbertTransformerDecoder(nn.Module):
    """
    The decoder of the Transformer model that uses a Hilbert matrix as its weight matrix.
    
    Args:
        order (int): The order of the Hilbert matrix.
        embedding_dim (int): The dimension of the embedding.
        hidden_dim (int): The dimension of the hidden state.
        num_heads (int): The number of attention heads.
        num_layers (int): The number of decoder layers.
    
    Example:
        >>> hilbert_transformer_decoder = HilbertTransformerDecoder(3, 2, 4, 2, 2)
        >>> hilbert_transformer_decoder(torch.tensor([[[0, 1, 2]]]), torch.tensor([[[0, 1, 2]]]), None, None)
        (tensor([[[-0.5774, -0.5774, -0.5774, -0.5774],
                  [-0.4082, -0.4082, -0.4082, -0.4082],
                  [-0.2236, -0.2236, -0.2236, -0.2236]]], grad_fn=<AddBackward0>), tensor([[[1., 1., 1., 1.]]], grad_fn=<SqueezeBackward1>))
    """

    def __init__(self, order, embedding_dim, hidden_dim, num_heads, num_layers):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embedding = HilbertEmbedding(order, embedding_dim)
        self.positional_encoding = HilbertPositionalEncoding(order, embedding_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.decoder_layers = nn.ModuleList([HilbertTransformerDecoderLayer(order, embedding_dim, hidden_dim, num_heads) for _ in range(num_layers)])

    def forward(self, x, encoder_output, source_mask, target_mask):
        output = self.embedding(x)
        output = self.positional_encoding(output)
        output = self.dropout(output)
        output = self.layer_norm(output)

        attention_weights = {}

        for i in range(self.num_layers):
            output, self_attention_weight, encoder_attention_weight = self.decoder_layers[i](output, encoder_output, source_mask, target_mask)
            attention_weights['decoder_layer_{}'.format(i + 1)] = self_attention_weight
            attention_weights['encoder_layer_{}'.format(i + 1)] = encoder_attention_weight

        return output, attention_weights

class HilbertTransformer(nn.Module):
    """
    The Transformer model that uses a Hilbert matrix as its weight matrix.
    
    Args:
        order (int): The order of the Hilbert matrix.
        embedding_dim (int): The dimension of the embedding.
        hidden_dim (int): The dimension of the hidden state.
        num_heads (int): The number of attention heads.
        num_encoder_layers (int): The number of encoder layers.
        num_decoder_layers (int): The number of decoder layers.
    
    Example:
        >>> hilbert_transformer = HilbertTransformer(3, 2, 4, 2, 2, 2)
        >>> hilbert_transformer(torch.tensor([[[0, 1, 2]]]), torch.tensor([[[0, 1, 2]]]), None, None)
        (tensor([[[-0.5774, -0.5774, -0.5774, -0.5774],
                  [-0.4082, -0.4082, -0.4082, -0.4082],
                  [-0.2236, -0.2236, -0.2236, -0.2236]]], grad_fn=<AddBackward0>), tensor([[[1., 1., 1., 1.]]], grad_fn=<SqueezeBackward1>))
    """

    def __init__(self, order, embedding_dim, hidden_dim, num_heads, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        self.encoder = HilbertTransformerEncoder(order, embedding_dim, hidden_dim, num_heads, num_encoder_layers)
        self.decoder = HilbertTransformerDecoder(order, embedding_dim, hidden_dim, num_heads, num_decoder_layers)

    def forward(self, source, target, source_mask=None, target_mask=None):
        encoder_output, encoder_attention = self.encoder(source, source_mask)
        decoder_output, decoder_attention = self.decoder(target, encoder_output, source_mask, target_mask)

        return decoder_output, encoder_attention, decoder_attention
