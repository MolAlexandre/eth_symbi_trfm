class ModelConfig:
    """Configuration du modèle BERT encoder-only"""
    
    def __init__(self, 
                 vocab_size=30522,
                 d_model=256,
                 num_heads=4,
                 num_layers=4,
                 d_hidden=1024,
                 max_len=128,
                 dropout=0.1,
                 symmetric_init=True):
        """
        Args:
            vocab_size: Taille du vocabulaire (30522 pour BERT tokenizer)
            d_model: Dimension des embeddings (256 pour mini modèle, 768 pour base)
            num_heads: Nombre de têtes d'attention (doit diviser d_model)
            num_layers: Nombre de couches encoder (4 pour expériences du papier)
            d_hidden: Dimension cachée du feedforward (4 * d_model généralement)
            max_len: Longueur maximale des séquences (128 pour dataset)
            dropout: Probabilité de dropout
            symmetric_init: True pour initialisation symétrique de Wqk
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_hidden = d_hidden
        self.max_len = max_len
        self.dropout = dropout
        self.symmetric_init = symmetric_init
        
        # Vérifications
        assert d_model % num_heads == 0, "d_model doit être divisible par num_heads"
        

class TrainingConfig:
    """Configuration de l'entraînement MLM"""
    
    def __init__(self,
                 batch_size=128,
                 gradient_accumulation_steps=2,
                 learning_rate=5e-5,
                 num_epochs=10,
                 warmup_ratio=0.1,
                 weight_decay=0.01,
                 gradient_clip=1.0,
                 mlm_probability=0.15,
                 num_workers = 1,
                 device='cuda',
                 mixed_precision=True):
        """
        Args:
            batch_size: Taille effective du batch (32 * 8 gradient accumulation = 256)
            learning_rate: Learning rate initial (5e-5 recommandé pour BERT)
            num_epochs: Nombre d'epochs d'entraînement
            warmup_ratio: Ratio steps de warmup linéaire (0.08 standard)
            weight_decay: Weight decay pour AdamW (0.01 standard)
            gradient_clip: Valeur max pour gradient clipping (1.0 standard)
            mlm_probability: Probabilité de masquage (15% standard BERT)
            device: 'cuda' ou 'cpu'
            mixed_precision: True pour fp16 mixed precision training
        """
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.effective_batch_size = batch_size * gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.mlm_probability = mlm_probability
        self.num_workers = num_workers
        self.device = device
        self.mixed_precision = mixed_precision


# Configurations pré-définies basées sur le papier
class BERTMiniConfig(ModelConfig):
    """Configuration BERT-Mini (4 layers) comme dans le papier"""
    def __init__(self, symmetric_init=False):
        super().__init__(
            vocab_size=30522,
            d_model=256,
            num_heads=4,
            num_layers=4,
            d_hidden=1024,
            max_len=128,
            dropout=0.1,
            symmetric_init=symmetric_init
        )


class BERTBaseConfig(ModelConfig):
    """Configuration BERT-Base (12 layers) comme dans le papier"""
    def __init__(self, symmetric_init=False):
        super().__init__(
            vocab_size=30522,
            d_model=768,
            num_heads=12,
            num_layers=12,
            d_hidden=3072,
            max_len=512,
            dropout=0.1,
            symmetric_init=symmetric_init
        )


class BERTLargeConfig(ModelConfig):
    """Configuration BERT-Large (24 layers) comme dans le papier"""
    def __init__(self, symmetric_init=False):
        super().__init__(
            vocab_size=30522,
            d_model=1024,
            num_heads=16,
            num_layers=24,
            d_hidden=4096,
            max_len=512,
            dropout=0.1,
            symmetric_init=symmetric_init
        )
