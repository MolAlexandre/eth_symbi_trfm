from datasets import load_dataset, DatasetDict, load_from_disk
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import os
from torchvision import datasets, transforms


class WikipediaDatasetManager:
    """
    Gère le chargement, la tokenization et la préparation du dataset Wikipedia pour MLM.
    Utilise un système de cache pour éviter de re-télécharger et re-processer les données.
    """
    
    def __init__(self, dataset_name="wikimedia/wikipedia", dataset_config="20231101.en", cache_dir="./data_cache"):
        """
        Args:
            dataset_name: Nom du dataset HuggingFace
            dataset_config: Configuration/snapshot du dataset
            cache_dir: Dossier racine pour tous les caches
        """
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.cache_dir = cache_dir
        
        # Initialiser le tokenizer une seule fois
        print("Initialisation du tokenizer BERT...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def load_and_split(self, train_size=1000000, val_size=100000, seed=45):
        """
        Charge le dataset Wikipedia et crée les splits train/validation.
        Utilise un cache pour éviter de re-télécharger et re-processer.
        
        Args:
            train_size: Nombre d'exemples pour l'entraînement
            val_size: Nombre d'exemples pour la validation
            seed: Seed pour reproductibilité du shuffle
            
        Returns:
            DatasetDict: {'train': Dataset, 'validation': Dataset}
        """
        # Nom du cache basé sur les paramètres
        cache_name = f"wiki_train{train_size}_val{val_size}_seed{seed}"
        cache_path = os.path.join(self.cache_dir, cache_name)
        
        # Vérifier si le cache existe
        if os.path.exists(cache_path):
            print(f"✓ Chargement du dataset depuis le cache: {cache_path}")
            ds = load_from_disk(cache_path)
            print(f"  Train: {len(ds['train'])} exemples")
            print(f"  Validation: {len(ds['validation'])} exemples")
            return ds
        
        # Sinon, télécharger et créer le dataset
        print(f"Cache non trouvé, téléchargement du dataset {self.dataset_name}...")
        dataset = load_dataset(
            self.dataset_name, 
            self.dataset_config, 
            split='train'
        )
        
        total_size = train_size + val_size
        print(f"Sélection de {total_size} exemples ({train_size} train, {val_size} val)...")
        
        # Shuffle et sélectionner total_size exemples
        shuffled = dataset.shuffle(seed=seed).select(range(total_size))
        
        # Créer les splits
        ds = DatasetDict({
            'train': shuffled.select(range(train_size)),
            'validation': shuffled.select(range(train_size, total_size))
        })
        
        # Sauvegarder dans le cache
        print(f"Sauvegarde du dataset dans le cache: {cache_path}")
        os.makedirs(self.cache_dir, exist_ok=True)
        ds.save_to_disk(cache_path)
        
        print(f"✓ Train: {len(ds['train'])} exemples")
        print(f"✓ Validation: {len(ds['validation'])} exemples")
        
        return ds
    
    def create_dataloaders(self, dataset_dict, batch_size=256, max_length=128, 
                          mlm_probability=0.15, num_workers=0):
        """
        Tokenize les datasets et crée les DataLoaders PyTorch avec masquage MLM.
        Utilise un cache pour éviter de re-tokeniser.
        
        Args:
            dataset_dict: DatasetDict retourné par load_and_split()
            batch_size: Taille du batch
            max_length: Longueur maximale des séquences (truncation)
            mlm_probability: Probabilité de masquage MLM (15% standard)
            num_workers: Nombre de workers pour DataLoader (0 = single-process)
            
        Returns:
            tuple: (train_dataloader, val_dataloader)
        """
        # Nom du cache pour les datasets tokenisés
        tokenized_cache = os.path.join(self.cache_dir, f"tokenized_maxlen{max_length}")
        
        # Vérifier si les datasets tokenisés existent
        if os.path.exists(tokenized_cache):
            print(f"✓ Chargement des datasets tokenisés depuis: {tokenized_cache}")
            tokenized_datasets = load_from_disk(tokenized_cache)
            train_dataset = tokenized_datasets['train']
            val_dataset = tokenized_datasets['validation']
        else:
            print("Cache de tokenization non trouvé, tokenization en cours...")
            
            # Fonction de tokenization à appliquer
            def tokenize_fn(examples):
                return self.tokenizer(
                    examples['text'],
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors=None
                )
            
            # Tokenization des datasets
            print("Tokenization du dataset train...")
            train_dataset = dataset_dict['train'].map(
                tokenize_fn,
                batched=True,
                remove_columns=dataset_dict['train'].column_names,
                desc="Tokenizing train"
            )
            
            print("Tokenization du dataset validation...")
            val_dataset = dataset_dict['validation'].map(
                tokenize_fn,
                batched=True,
                remove_columns=dataset_dict['validation'].column_names,
                desc="Tokenizing validation"
            )
            
            # Sauvegarder les datasets tokenisés
            print(f"Sauvegarde des datasets tokenisés: {tokenized_cache}")
            os.makedirs(self.cache_dir, exist_ok=True)
            tokenized_datasets = DatasetDict({
                'train': train_dataset,
                'validation': val_dataset
            })
            tokenized_datasets.save_to_disk(tokenized_cache)
            print("✓ Datasets tokenisés sauvegardés")
        
        # Convertir en format PyTorch
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        
        # Créer le collator MLM
        collator = MLMCollator(
            tokenizer=self.tokenizer,
            mlm_probability=mlm_probability
        )
        
        # Créer les DataLoaders PyTorch
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        print(f"✓ DataLoaders créés: {len(train_loader)} batches train, {len(val_loader)} batches val")
        
        return train_loader, val_loader


class MLMCollator:
    """
    Data collator pour Masked Language Modeling (MLM).
    Applique dynamiquement la stratégie de masquage BERT (80-10-10) à chaque batch.
    """
    
    def __init__(self, tokenizer, mlm_probability=0.15):
        """
        Args:
            tokenizer: BertTokenizer avec vocabulaire et tokens spéciaux
            mlm_probability: Probabilité de masquer un token (0.15 = 15%)
        """
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        
        # Cache des IDs de tokens spéciaux
        self.mask_token_id = tokenizer.mask_token_id  # [MASK]
        self.pad_token_id = tokenizer.pad_token_id    # [PAD]
        self.cls_token_id = tokenizer.cls_token_id    # [CLS]
        self.sep_token_id = tokenizer.sep_token_id    # [SEP]
        self.vocab_size = tokenizer.vocab_size
    
    def __call__(self, batch):
        """
        Appelé automatiquement par DataLoader pour assembler et masquer chaque batch.
        
        Args:
            batch: Liste de dicts [{'input_ids': tensor, 'attention_mask': tensor}, ...]
            
        Returns:
            dict: {'input_ids': tensor, 'labels': tensor, 'attention_mask': tensor}
        """
        # Empiler tous les tensors du batch
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        # Cloner input_ids AVANT masquage pour créer les labels
        labels = input_ids.clone()
        
        # Appliquer le masquage MLM
        input_ids, labels = self.mask_tokens(input_ids, labels)
        
        return {
            'input_ids': input_ids,          # Séquence avec tokens masqués
            'labels': labels,                # Vraie séquence (-100 pour non-masqués)
            'attention_mask': attention_mask # Masque de padding
        }
    
    def mask_tokens(self, input_ids, labels):
        """
        Applique la stratégie MLM de BERT sur 15% des tokens (hors spéciaux):
        - 80% → remplacés par [MASK]
        - 10% → remplacés par un token aléatoire
        - 10% → inchangés (mais supervisés)
        
        Args:
            input_ids: Tensor (batch_size, seq_len) - séquence originale
            labels: Tensor (batch_size, seq_len) - copie de input_ids
            
        Returns:
            tuple: (input_ids masqués, labels avec -100 pour non-masqués)
        """
        # 1. Créer tenseur de probabilité (15% partout)
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
        
        # 2. Exclure tokens spéciaux ([PAD], [CLS], [SEP]) et 0 pour token spéciaux
        special_tokens_mask = self._get_special_tokens_mask(input_ids)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # 3. Tirer aléatoirement les tokens à masquer (~15%)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 4. Mettre -100 dans labels pour positions NON masquées
        # (ignoré par CrossEntropyLoss)
        labels[~masked_indices] = -100
        
        # 5. Parmi les tokens masqués: 80% → [MASK]
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id
        
        # 6. Parmi les restants: 50% (= 10% du total) → random token
        indices_random = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() 
            & masked_indices 
            & ~indices_replaced
        )
        random_words = torch.randint(self.vocab_size, input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        # 7. Les 10% restants restent inchangés (déjà dans input_ids)
        
        return input_ids, labels
    
    def _get_special_tokens_mask(self, input_ids):
        """
        Crée un masque booléen identifiant les tokens spéciaux.
        
        Args:
            input_ids: Tensor (batch_size, seq_len)
            
        Returns:
            Tensor bool: True pour [PAD], [CLS], [SEP]
        """
        return (
            (input_ids == self.pad_token_id) |
            (input_ids == self.cls_token_id) |
            (input_ids == self.sep_token_id)
        )
    

class CifarDatasetManager:
    def __init__(self, cache_dir="./data_cache"):
        self.cache_dir = cache_dir
        self.cifar_cache = os.path.join(cache_dir, "cifar10")

        self.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616))
            ])
        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616))
            ])

    def create_dataloaders(self, batch_size, num_workers):
        # Vérifier si CIFAR-10 est déjà téléchargé
        cifar_exists = os.path.exists(os.path.join(self.cifar_cache, "cifar-10-batches-py"))
        
        if cifar_exists:
            print(f"✓ Chargement de CIFAR-10 depuis le cache: {self.cifar_cache}")
        else:
            print(f"Cache non trouvé, téléchargement de CIFAR-10...")
            os.makedirs(self.cifar_cache, exist_ok=True)
        
        train_dataset = datasets.CIFAR10(
            root=self.cifar_cache,
            train=True,
            download=not cifar_exists,
            transform=self.transform_train
        )

        val_dataset = datasets.CIFAR10(
            root=self.cifar_cache,
            train=False,
            download=not cifar_exists,
            transform=self.transform_val
        )

        if not cifar_exists:
            print("✓ CIFAR-10 téléchargé et sauvegardé dans le cache")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        print(f"✓ DataLoaders créés: {len(train_loader)} batches train, {len(val_loader)} batches val")
        print(f"  Train: {len(train_dataset)} exemples")
        print(f"  Validation: {len(val_dataset)} exemples")

        return train_loader, val_loader