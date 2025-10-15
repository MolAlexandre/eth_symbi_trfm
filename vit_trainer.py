import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import csv
import os
from datetime import datetime
from symmetry_score import compute_model_symmetry

class VitTrainer:
    """Gère l'entraînement et la validation du modèle ViT."""

    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Créer le dossier de sauvegarde avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = "symmetric" if getattr(config, 'symmetric_init', False) else "standard"
        self.save_dir = f"checkpoints/{timestamp}_{model_type}"
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"✓ Dossier de sauvegarde: {self.save_dir}")

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

        # Loss function
        label_smoothing = getattr(config, 'label_smoothing', 0.0)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # État de l'entraînement
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0
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
        total_correct = 0
        total_samples = 0
        num_batches = 0
        accumulation_counter = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", dynamic_ncols=True)

        for batch in pbar:
            # Gérer les deux formats possibles de batch
            if isinstance(batch, dict):
                images = batch['image'].to(self.config.device)
                labels = batch['label'].to(self.config.device)
            else:
                # Format tuple/liste (images, labels)
                images, labels = batch
                images = images.to(self.config.device)
                labels = labels.to(self.config.device)

            # Forward pass
            if self.config.mixed_precision:
                with autocast(device_type="cuda"):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            # Backward pass
            loss = loss / self.config.gradient_accumulation_steps

            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulation_counter += 1

            # Mise à jour des poids
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
                self.global_step += 1

            # Calculer accuracy
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            # Progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{loss.item() * self.config.gradient_accumulation_steps:.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}',
                'acc': f'{100.0 * total_correct / total_samples:.2f}%',
                'lr': f'{current_lr:.2e}',
                'step': self.global_step
            })

        avg_loss = total_loss / num_batches
        avg_acc = 100.0 * total_correct / total_samples
        return avg_loss, avg_acc

    def validate(self):
        """Évalue le modèle sur le validation set."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", dynamic_ncols=True)

            for batch in pbar:
                # Gérer les deux formats possibles de batch
                if isinstance(batch, dict):
                    images = batch['image'].to(self.config.device)
                    labels = batch['label'].to(self.config.device)
                else:
                    # Format tuple/liste (images, labels)
                    images, labels = batch
                    images = images.to(self.config.device)
                    labels = labels.to(self.config.device)

                # Forward pass
                if self.config.mixed_precision:
                    with autocast(device_type='cuda'):
                        logits = self.model(images)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                # Calculer accuracy
                preds = torch.argmax(logits, dim=-1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({
                    'val_loss': f'{total_loss/num_batches:.4f}',
                    'val_acc': f'{100.0 * total_correct / total_samples:.2f}%'
                })

        avg_loss = total_loss / num_batches
        avg_acc = 100.0 * total_correct / total_samples
        return avg_loss, avg_acc

    def save_checkpoint(self, epoch, val_acc, checkpoint_path):
        """
        Sauvegarde un checkpoint avec TOUT L'HISTORIQUE des métriques.
        Comme dans BERT: metrics_history contient toutes les epochs précédentes.
        """
        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'metrics_history': self.metrics_history  # TOUT L'HISTORIQUE ICI
        }, checkpoint_path)
        print(f"✓ Checkpoint sauvegardé: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """
        Charge un checkpoint et RÉCUPÈRE tout l'historique des métriques.
        """
        print(f"Chargement du checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.metrics_history = checkpoint.get('metrics_history', [])

        print(f"✓ Reprise à partir de l'epoch {self.start_epoch}")
        print(f"✓ Global step: {self.global_step}")
        print(f"✓ Meilleure val acc: {self.best_val_acc:.2f}%")
        print(f"✓ {len(self.metrics_history)} epochs d'historique récupérées")

    def save_metrics_csv(self, num_layers):
        """
        Sauvegarde TOUTES les métriques de metrics_history dans un CSV.
        Peut être appelé à tout moment pour regénérer le CSV depuis le checkpoint.
        """
        if not self.metrics_history:
            print("⚠ Pas de métriques à sauvegarder")
            return

        csv_path = os.path.join(self.save_dir, "metrics.csv")

        # Fieldnames basés sur la première entrée
        fieldnames = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'symmetry_avg']
        for i in range(num_layers):
            fieldnames.append(f'layer_{i}')

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metrics_history)

        print(f"✓ Métriques sauvegardées: {csv_path} ({len(self.metrics_history)} epochs)")

    def train(self, num_epochs, num_layers):
        """Boucle d'entraînement complète."""
        print(f"\nDébut de l'entraînement (epoch {self.start_epoch+1} -> {num_epochs})...")
        print("="*80)

        for epoch in range(self.start_epoch, num_epochs):
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch+1}/{num_epochs}")
            print(f"{'='*80}")

            # Entraînement
            train_loss, train_acc = self.train_epoch(epoch)
            print(f"\n[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

            # Validation
            val_loss, val_acc = self.validate()
            print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            # Calculer les scores de symétrie
            symmetry_scores = compute_model_symmetry(self.model)
            print(f"[Epoch {epoch+1}] Symmetry Avg: {symmetry_scores['average']:.4f}")

            # AJOUTER À L'HISTORIQUE
            self.metrics_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'symmetry_avg': symmetry_scores['average'],
                **{k: v for k, v in symmetry_scores.items() if k != 'average'}
            })

            # Sauvegarder checkpoint de l'epoch (avec tout l'historique)
            if (epoch + 1) % 25 == 0:
                checkpoint_path = os.path.join(self.save_dir, f"epoch_{epoch+1}.pt")
                self.save_checkpoint(epoch, val_acc, checkpoint_path)
                print(f"✓ Checkpoint sauvegardé pour l'epoch {epoch+1}")

            # Sauvegarder le meilleur modèle (avec tout l'historique)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                best_path = os.path.join(self.save_dir, "best.pt")
                self.save_checkpoint(epoch, val_acc, best_path)
                print(f"✓ Nouveau meilleur modèle! (val_acc={val_acc:.2f}%)")

            # Sauvegarder le CSV (peut être regénéré depuis n'importe quel checkpoint)
            self.save_metrics_csv(num_layers)

        print("\n" + "="*80)
        print("ENTRAÎNEMENT TERMINÉ!")
        print(f"Meilleure val acc: {self.best_val_acc:.2f}%")
        print(f"Tous les fichiers sont dans: {self.save_dir}")
        print("="*80)

        return self.best_val_acc
