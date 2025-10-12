
import torch
import torch.nn as nn
import math
from embedding import BERTEmbeddings

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, symmetric_init=False):
        '''
        Multi-head attention with optional symmetric Wqk initialization.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            symmetric_init: If True, initialize Wq and Wk such that Wqk = Wq @ Wk.T is symmetric
        '''
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads # Dimension per head
        self.symmetric_init = symmetric_init

        # Query, Key, Value projections (modules)
        self.query = nn.Linear(d_model, d_model, bias=True)
        self.key = nn.Linear(d_model, d_model, bias=True)
        self.value = nn.Linear(d_model, d_model, bias=True)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with optional symmetric constraint on Wqk."""
        if self.symmetric_init:
            # Wq initialisé aléatoirement, puis Wk = Wq
            with torch.no_grad():
                # Initialisation standard de Wq
                nn.init.xavier_uniform_(self.query.weight)
                nn.init.zeros_(self.query.bias)
                
                # Copie exacte : Wk = Wq
                # Ceci garantit que Wqk = Wq @ Wk.T = Wq @ Wq.T (symétrique)
                self.key.weight.copy_(self.query.weight)
                self.key.bias.copy_(self.query.bias)
        else:
            # Initialisation standard (non-symétrique)
            nn.init.xavier_uniform_(self.query.weight)
            nn.init.xavier_uniform_(self.key.weight)
            nn.init.zeros_(self.query.bias)
            nn.init.zeros_(self.key.bias)
        
        # Initialisation standard pour V et output
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.zeros_(self.value.bias)
        nn.init.zeros_(self.W_o.bias)

    def forward(self, x, mask=None):
        '''
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]

        Returns:
            output: [batch_size, seq_len, d_model]
        '''
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attention_output = torch.matmul(attention_weights, V)

        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.W_o(attention_output)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.1):
        '''
        Position-wise feed-forward network.

        Args:
            d_model: Model dimension
            d_hidden: Hidden dimension
            dropout: Dropout probability
        '''
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_hidden, dropout=0.1, symmetric_init=False):
        '''
        Encoder layer (attention + FFN).

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_hidden: Feed-forward hidden dimension
            dropout: Dropout probability
            symmetric_init: Use symmetric Wqk initialization
        '''
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout, symmetric_init)
        self.feed_forward = FeedForward(d_model, d_hidden, dropout)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-12)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_hidden, 
                 max_len=512, dropout=0.1, symmetric_init=False):
        '''
        Complete encoder-only Transformer.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            d_hidden: Feed-forward hidden dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
            symmetric_init: Use symmetric Wqk initialization for all layers
        '''
        super().__init__()

        self.d_model = d_model
        self.symmetric_init = symmetric_init

        # Use the existing BERTEmbeddings from bert.py
        self.embeddings = BERTEmbeddings(vocab_size, d_model, max_len, dropout)

        # Encoder layers
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_hidden, dropout, symmetric_init)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        '''
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            output: [batch_size, seq_len, d_model]
        '''
        # Embeddings
        x = self.embeddings(input_ids)

        # Convert attention mask to the right shape if provided
        if attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        return x


class MLMHead(nn.Module):
    """
    Tête de prédiction pour Masked Language Modeling (MLM).
    Transforme les sorties de l'encoder (d_model) vers l'espace vocabulaire.
    """
    
    def __init__(self, d_model, vocab_size):
        """
        Args:
            d_model: Dimension des embeddings de sortie de l'encoder
            vocab_size: Taille du vocabulaire pour la prédiction
        """
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.activation = nn.GELU()
        self.dense = nn.Linear(d_model, d_model)
        self.decoder = nn.Linear(d_model, vocab_size, bias=True)
        
    def forward(self, x):
        """
        Args:
            x: Tensor (batch_size, seq_len, d_model)
            
        Returns:
            logits: Tensor (batch_size, seq_len, vocab_size)
        """
        # Transformation dense + activation
        x = self.dense(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        
        # Projection vers vocabulaire
        logits = self.decoder(x)
        
        return logits


class BERTForMLM(nn.Module):
    """
    Modèle BERT complet pour Masked Language Modeling.
    Combine TransformerEncoder + MLMHead.
    """
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_hidden,
                 max_len=512, dropout=0.1, symmetric_init=False):
        """
        Args:
            vocab_size: Taille du vocabulaire
            d_model: Dimension du modèle
            num_heads: Nombre de têtes d'attention
            num_layers: Nombre de couches encoder
            d_hidden: Dimension cachée du feedforward
            max_len: Longueur maximale des séquences
            dropout: Probabilité de dropout
            symmetric_init: Si True, initialise Wq = Wk pour symétrie de Wqk
        """
        super().__init__()
        
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_hidden=d_hidden,
            max_len=max_len,
            dropout=dropout,
            symmetric_init=symmetric_init
        )
        
        self.mlm_head = MLMHead(d_model, vocab_size)
        
        self.vocab_size = vocab_size
        
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: Tensor (batch_size, seq_len) - indices des tokens
            attention_mask: Tensor (batch_size, seq_len) - masque d'attention
            
        Returns:
            logits: Tensor (batch_size, seq_len, vocab_size)
        """
        # Passage dans l'encoder
        encoder_output = self.encoder(input_ids, attention_mask)
        logits = self.mlm_head(encoder_output)
        
        return logits
    
    def compute_loss(self, input_ids, labels, attention_mask=None):
        """
        Calcule la loss MLM uniquement sur les tokens masqués.
        
        Args:
            input_ids: Tensor (batch_size, seq_len)
            labels: Tensor (batch_size, seq_len) - labels avec -100 pour tokens non-masqués
            attention_mask: Tensor (batch_size, seq_len)
            
        Returns:
            loss: Scalar tensor
        """
        # Forward pass
        logits = self.forward(input_ids, attention_mask)
        
        # Reshape pour cross-entropy
        # logits: (batch_size * seq_len, vocab_size)
        # labels: (batch_size * seq_len)
        logits_flat = logits.view(-1, self.vocab_size)
        labels_flat = labels.view(-1)
        
        # Cross-entropy loss (ignore_index=-100 pour tokens non-masqués)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits_flat, labels_flat)
        
        return loss
