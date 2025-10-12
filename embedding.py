import torch
import torch.nn as nn
import math

class BERTEmbeddings(nn.Module):   
    def __init__(self, vocab_size, d_model, max_len=512, dropout=0.1):
        """
        Args:
            vocab_size: Taille du vocabulaire (30522 pour BERT)
            d_model: Dimension du modèle / dimension d'un token (256 pour votre config)
            max_len: Longueur maximale des séquences (128 pour go_emotions)
            dropout: Taux de dropout
        """
        super().__init__()
        
        # Token embedding (apprenable)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Position embedding (fixe, non apprenable)
        pe = self._create_positional_encoding(max_len, d_model)
        self.register_buffer('positional_encoding', pe)
        
        # LayerNorm (eps=1e-12 comme BERT original)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def _create_positional_encoding(self, max_len, d_model):

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * - (math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, max_len, d_model]
    
    def forward(self, input_ids):

        seq_len = input_ids.size(1)
        
        # Token embeddings (apprenable)
        token_emb = self.token_embedding(input_ids)
        
        # Position embeddings (fixe)
        position_emb = self.positional_encoding[:, :seq_len, :]
        
        # Addition token + position
        embeddings = token_emb + position_emb
        
        # LayerNorm + Dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class VITEmbeddings(nn.Module):
    def __init__(self):
        pass

    def _create_positional_encoding(self):
        pass

    
    def forward(self, x):
        pass


