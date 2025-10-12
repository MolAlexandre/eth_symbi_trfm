import torch
import argparse
import os
from config import BERTMiniConfig, TrainingConfig
from model import BERTForMLM
from data_loader import WikipediaDatasetManager
from trainer import Trainer

def train_model(model_name, resume_checkpoint=None):
    """
    Entraîne un modèle.
    
    Args:
        model_name: Nom du modèle (ex: 'standard' ou 'symmetric')
        resume_checkpoint: Chemin vers un checkpoint pour reprendre l'entraînement
    """
    
    print("="*80)
    print(f"ENTRAÎNEMENT - Modèle: {model_name.upper()}")
    print("="*80)
    
    # Configuration
    symmetric_init = (model_name == "symmetric")
    model_config = BERTMiniConfig(symmetric_init=symmetric_init)
    train_config = TrainingConfig()
    
    print(f"✓ Initialisation: {'SYMMETRIC' if symmetric_init else 'STANDARD'}")
    print(f"✓ Device: {train_config.device}")
    print(f"✓ Batch effectif: {train_config.effective_batch_size}")
    
    # Chargement des données
    print("\nChargement des données...")
    data_manager = WikipediaDatasetManager()
    dataset = data_manager.load_and_split()
    train_loader, val_loader = data_manager.create_dataloaders(
        dataset,
        batch_size=train_config.batch_size,
        max_length=model_config.max_len,
        mlm_probability=train_config.mlm_probability,
        num_workers=train_config.num_workers
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Création du modèle
    print("\nCréation du modèle...")
    model = BERTForMLM(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        num_heads=model_config.num_heads,
        num_layers=model_config.num_layers,
        d_hidden=model_config.d_hidden,
        max_len=model_config.max_len,
        dropout=model_config.dropout,
        symmetric_init=model_config.symmetric_init
    )
    model = model.to(train_config.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Paramètres: {num_params:,}")
    
    # Créer le trainer
    trainer = Trainer(model, train_loader, val_loader, train_config)
    
    # Charger un checkpoint si spécifié
    if resume_checkpoint:
        if os.path.exists(resume_checkpoint):
            trainer.load_checkpoint(resume_checkpoint)
        else:
            print(f"Checkpoint non trouvé: {resume_checkpoint}")
            print("Démarrage d'un nouvel entraînement...")
    
    # Lancer l'entraînement
    best_val_loss = trainer.train(
        num_epochs=train_config.num_epochs,
        checkpoint_prefix=model_name,
        num_layers=model_config.num_layers
    )
    
    return best_val_loss

def main():
    parser = argparse.ArgumentParser(description='Entraînement BERT avec/sans initialisation symétrique')
    parser.add_argument('--model', type=str, default='standard', 
                       choices=['standard', 'symmetric'],
                       help='Type de modèle à entraîner')
    parser.add_argument('--resume', type=str, default=None,
                       help='Chemin vers un checkpoint pour reprendre l\'entraînement')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(f"EXPÉRIENCE: ENTRAÎNEMENT {args.model.upper()}")
    print("="*80)
    
    train_model(args.model, args.resume)
    
    print("\n" + "="*80)
    print("ENTRAÎNEMENT TERMINÉ!")
    print(f"Fichiers générés:")
    print(f"  - metrics_{args.model}.csv")
    print(f"  - checkpoints/{args.model}_best.pt")
    print(f"  - checkpoints/{args.model}_epoch_*.pt")
    print("="*80)

if __name__ == "__main__":
    main()