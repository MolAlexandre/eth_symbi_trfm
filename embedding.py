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
    def __init__(self, d_embedding: int, patch_size: int, img_size: int = 32, in_channels: int = 3):
        super().__init__()
        self.d_embedding = d_embedding
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Projection des patches avec Conv2d
        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=d_embedding, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # CLS token: un seul token appris, étendu dynamiquement
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_embedding))
        
        # Position embedding: num_patches + 1 pour le CLS token
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, d_embedding) * 0.02)

        self.layer_norm = nn.LayerNorm(d_embedding, eps=1e-6)
        self.dropout = nn.Dropout(0.0)  # Standard ViT
    
    def forward(self, x):
        # x shape: [batch_size, 3, 32, 32]
        batch_size = x.shape[0]
        
        # Projection des patches
        x = self.patch_embedding(x)  # [batch_size, d_embedding, h_patches, w_patches]
        x = x.flatten(2)  # [batch_size, d_embedding, num_patches]
        x = x.transpose(1, 2)  # [batch_size, num_patches, d_embedding]
        
        # Expansion du CLS token pour le batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, d_embedding]
        
        # Concaténation CLS + patches
        x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, num_patches + 1, d_embedding]
        
        # Ajout des positional embeddings
        x = x + self.position_embedding  # [batch_size, num_patches + 1, d_embedding]
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return x
