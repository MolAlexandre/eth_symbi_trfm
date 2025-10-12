import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import csv
import os

from symmetry_score import compute_model_symmetry, log_symmetry_scores

class Trainer:
    """Gère l'entraînement et la validation du modèle."""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Calculer les steps
        self.steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
        self.total_steps = self.steps_per_epoch * config.num_epochs
        self.warmup_steps = int(self.total_steps * config.warmup_ratio)
        
        # Initialiser optimizer, scheduler, scaler
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.scheduler = self._get_linear_warmup_scheduler()
        self.scaler = GradScaler(enabled=config.mixed_precision)
        
        # État de l'entraînement
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.metrics_history = []
    
    def _get_linear_warmup_scheduler(self):
        """Crée un scheduler avec warmup linéaire puis décroissance linéaire."""
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return max(0.0, float(self.total_steps - current_step) / 
                      float(max(1, self.total_steps - self.warmup_steps)))
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, epoch):
        """Entraîne le modèle pour une epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        accumulation_counter = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", dynamic_ncols=True)
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.config.device)
            labels = batch['labels'].to(self.config.device)
            attention_mask = batch['attention_mask'].to(self.config.device)
            
            if self.config.mixed_precision:
                with autocast(device_type="cuda"):
                    loss = self.model.compute_loss(input_ids, labels, attention_mask)
            else:
                loss = self.model.compute_loss(input_ids, labels, attention_mask)
            
            loss = loss / self.config.gradient_accumulation_steps
            
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulation_counter += 1
            
            if accumulation_counter % self.config.gradient_accumulation_steps == 0:
                if self.config.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                
                if self.config.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                accumulation_counter = 0
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            self.global_step += 1
            
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{loss.item() * self.config.gradient_accumulation_steps:.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{current_lr:.2e}',
                'step': self.global_step
            })
        
        return total_loss / num_batches
    
    def validate(self):
        """Évalue le modèle sur le validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", dynamic_ncols=True)
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                
                if self.config.mixed_precision:
                    with autocast(device_type='cuda'):
                        loss = self.model.compute_loss(input_ids, labels, attention_mask)
                else:
                    loss = self.model.compute_loss(input_ids, labels, attention_mask)
                
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'val_loss': f'{total_loss/num_batches:.4f}'})
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, perplexity
    
    def save_checkpoint(self, epoch, val_loss, checkpoint_path):
        """Sauvegarde un checkpoint."""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'metrics_history': self.metrics_history
        }, checkpoint_path)
        
        print(f"✓ Checkpoint sauvegardé: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Charge un checkpoint."""
        print(f"Chargement du checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.metrics_history = checkpoint.get('metrics_history', [])
        
        print(f"✓ Reprise à partir de l'epoch {self.start_epoch}")
        print(f"✓ Global step: {self.global_step}")
        print(f"✓ Meilleure val loss: {self.best_val_loss:.4f}")
    
    def save_metrics_csv(self, csv_path, num_layers):
        """Sauvegarde les métriques dans un CSV."""
        if not self.metrics_history:
            return
        
        fieldnames = ['epoch', 'train_loss', 'val_loss', 'perplexity', 'symmetry_avg']
        for i in range(num_layers):
            fieldnames.append(f'layer_{i}')
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metrics_history)
        
        print(f"✓ Métriques sauvegardées: {csv_path}")
    
    def train(self, num_epochs, checkpoint_prefix, num_layers):
        """Boucle d'entraînement complète."""
        print(f"\nDébut de l'entraînement (epoch {self.start_epoch+1} -> {num_epochs})...")
        print("="*80)
        
        os.makedirs("checkpoints", exist_ok=True)
        
        for epoch in range(self.start_epoch, num_epochs):
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch+1}/{num_epochs}")
            print(f"{'='*80}")
            
            # Entraînement
            train_loss = self.train_epoch(epoch)
            print(f"\n[Epoch {epoch+1}] Train Loss: {train_loss:.4f}")
            
            # Validation
            val_loss, perplexity = self.validate()
            print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f} | Perplexity: {perplexity:.2f}")
            
            # Calcul de symétrie
            symmetry_scores = compute_model_symmetry(self.model)
            log_symmetry_scores(symmetry_scores, epoch)
            
            # Sauvegarder les métriques
            self.metrics_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'perplexity': perplexity,
                'symmetry_avg': symmetry_scores['average'],
                **{k: v for k, v in symmetry_scores.items() if k != 'average'}
            })
            
            # Sauvegarder checkpoint de l'epoch
            checkpoint_path = f"checkpoints/{checkpoint_prefix}_epoch_{epoch+1}.pt"
            self.save_checkpoint(epoch, val_loss, checkpoint_path)
            
            # Sauvegarder le meilleur modèle
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = f"checkpoints/{checkpoint_prefix}_best.pt"
                self.save_checkpoint(epoch, val_loss, best_path)
                print(f"✓ Nouveau meilleur modèle! (val_loss={val_loss:.4f})")
            
            # Sauvegarder les métriques CSV
            csv_path = f"metrics_{checkpoint_prefix}.csv"
            self.save_metrics_csv(csv_path, num_layers)
        
        print("\n" + "="*80)
        print("ENTRAÎNEMENT TERMINÉ!")
        print(f"Meilleure val loss: {self.best_val_loss:.4f}")
        print("="*80)
        
        return self.best_val_loss