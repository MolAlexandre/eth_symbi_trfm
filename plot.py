import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_convergence_comparison(csv_standard, csv_symmetric, save_path='convergence_comparison.png'):
    """Compare la convergence entre init standard et symétrique."""
    df_std = pd.read_csv(csv_standard)
    df_sym = pd.read_csv(csv_symmetric)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # 1. Validation Loss (métrique PRINCIPALE pour convergence)
    ax1 = axes[0, 0]
    ax1.plot(df_std['epoch'], df_std['val_loss'], 
             label='Standard Init', linewidth=2.5, marker='o', markersize=6, color='#1f77b4')
    ax1.plot(df_sym['epoch'], df_sym['val_loss'],
             label='Symmetric Init', linewidth=2.5, marker='s', markersize=6, color='#ff7f0e')
    ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Validation Loss', fontsize=13, fontweight='bold')
    ax1.set_title('Validation Loss: Convergence Comparison\n(Métrique principale)', 
                  fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12, loc='upper right')
    ax1.grid(alpha=0.3, linestyle='--')

    # 2. Training Loss
    ax2 = axes[0, 1]
    ax2.plot(df_std['epoch'], df_std['train_loss'],
             label='Standard Init', linewidth=2.5, marker='o', markersize=6, color='#1f77b4')
    ax2.plot(df_sym['epoch'], df_sym['train_loss'],
             label='Symmetric Init', linewidth=2.5, marker='s', markersize=6, color='#ff7f0e')
    ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Training Loss', fontsize=13, fontweight='bold')
    ax2.set_title('Training Loss Evolution', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=12, loc='upper right')
    ax2.grid(alpha=0.3, linestyle='--')

    # 3. Perplexity (sur échelle log si nécessaire)
    ax3 = axes[1, 0]
    ax3.plot(df_std['epoch'], df_std['perplexity'],
             label='Standard Init', linewidth=2.5, marker='o', markersize=6, color='#1f77b4')
    ax3.plot(df_sym['epoch'], df_sym['perplexity'],
             label='Symmetric Init', linewidth=2.5, marker='s', markersize=6, color='#ff7f0e')
    ax3.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Perplexity', fontsize=13, fontweight='bold')
    ax3.set_title('Perplexity Evolution', fontsize=15, fontweight='bold')
    ax3.legend(fontsize=12, loc='upper right')
    ax3.grid(alpha=0.3, linestyle='--')
    # Option: échelle log si perplexity très élevée
    # ax3.set_yscale('log')

    # 4. Symmetry Score Evolution (NOUVEAU: ce qui manquait!)
    ax4 = axes[1, 1]
    ax4.plot(df_std['epoch'], df_std['symmetry_avg'],
             label='Standard Init', linewidth=2.5, marker='o', markersize=6, color='#1f77b4')
    ax4.plot(df_sym['epoch'], df_sym['symmetry_avg'],
             label='Symmetric Init', linewidth=2.5, marker='s', markersize=6, color='#ff7f0e')
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Baseline')
    ax4.axhline(y=1, color='green', linestyle='--', linewidth=1.5, alpha=0.4, label='Perfect Symmetry')
    ax4.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Avg Symmetry Score', fontsize=13, fontweight='bold')
    ax4.set_title('Wqk Symmetry During Training\n(Papier: Theorem 2.4)', 
                  fontsize=15, fontweight='bold')
    ax4.legend(fontsize=11, loc='lower right')
    ax4.grid(alpha=0.3, linestyle='--')
    ax4.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé: {save_path}")

    # Calcul du speedup (CORRIGÉ)
    print(f"\n{'='*70}")
    print(f"RÉSULTATS DE L\'EXPÉRIENCE")
    print(f"{'='*70}")

    final_loss_std = df_std['val_loss'].iloc[-1]
    final_loss_sym = df_sym['val_loss'].iloc[-1]

    # Trouver à quelle epoch symmetric atteint la loss finale de standard
    epochs_sym_match = df_sym[df_sym['val_loss'] <= final_loss_std]['epoch'].values

    if len(epochs_sym_match) > 0:
        epoch_match = epochs_sym_match[0]
        speedup = ((df_std['epoch'].iloc[-1] - epoch_match) / df_std['epoch'].iloc[-1]) * 100
        print(f"Loss finale (Standard):   {final_loss_std:.4f} @ epoch {df_std['epoch'].iloc[-1]}")
        print(f"Loss finale (Symmetric):  {final_loss_sym:.4f} @ epoch {df_sym['epoch'].iloc[-1]}")
        print(f"\nSymmetric atteint {final_loss_std:.4f} à l\'epoch {epoch_match}")
        print(f"Convergence speedup:      {speedup:.1f}%")
    else:
        print(f"Loss finale (Standard):   {final_loss_std:.4f}")
        print(f"Loss finale (Symmetric):  {final_loss_sym:.4f}")
        print(f"Amélioration:             {((final_loss_std - final_loss_sym)/final_loss_std)*100:.1f}%")

    print(f"\nSymétrie finale:")
    print(f"  Standard:   {df_std['symmetry_avg'].iloc[-1]:.4f}")
    print(f"  Symmetric:  {df_sym['symmetry_avg'].iloc[-1]:.4f}")
    print(f"{'='*70}\n")


def plot_convergence_one(csv_standard, save_path='result_1M_100K_standard.png'):
    """Compare la convergence entre init standard et symétrique."""
    df_std = pd.read_csv(csv_standard)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # 1. Validation Loss (métrique PRINCIPALE pour convergence)
    ax1 = axes[0, 0]
    ax1.plot(df_std['epoch'], df_std['val_loss'], 
             label='Standard Init', linewidth=2.5, marker='o', markersize=6, color='#1f77b4')
    ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Validation Loss', fontsize=13, fontweight='bold')
    ax1.set_title('Validation Loss: Convergence Comparison\n(Métrique principale)', 
                  fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12, loc='upper right')
    ax1.grid(alpha=0.3, linestyle='--')

    # 2. Training Loss
    ax2 = axes[0, 1]
    ax2.plot(df_std['epoch'], df_std['train_loss'],
             label='Standard Init', linewidth=2.5, marker='o', markersize=6, color='#1f77b4')
    ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Training Loss', fontsize=13, fontweight='bold')
    ax2.set_title('Training Loss Evolution', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=12, loc='upper right')
    ax2.grid(alpha=0.3, linestyle='--')

    # 3. Perplexity (sur échelle log si nécessaire)
    ax3 = axes[1, 0]
    ax3.plot(df_std['epoch'], df_std['perplexity'],
             label='Standard Init', linewidth=2.5, marker='o', markersize=6, color='#1f77b4')
    ax3.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Perplexity', fontsize=13, fontweight='bold')
    ax3.set_title('Perplexity Evolution', fontsize=15, fontweight='bold')
    ax3.legend(fontsize=12, loc='upper right')
    ax3.grid(alpha=0.3, linestyle='--')
    # Option: échelle log si perplexity très élevée
    # ax3.set_yscale('log')

    # 4. Symmetry Score Evolution (NOUVEAU: ce qui manquait!)
    ax4 = axes[1, 1]
    ax4.plot(df_std['epoch'], df_std['symmetry_avg'],
             label='Standard Init', linewidth=2.5, marker='o', markersize=6, color='#1f77b4')
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Baseline')
    ax4.axhline(y=1, color='green', linestyle='--', linewidth=1.5, alpha=0.4, label='Perfect Symmetry')
    ax4.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Avg Symmetry Score', fontsize=13, fontweight='bold')
    ax4.set_title('Wqk Symmetry During Training\n(Papier: Theorem 2.4)', 
                  fontsize=15, fontweight='bold')
    ax4.legend(fontsize=11, loc='lower right')
    ax4.grid(alpha=0.3, linestyle='--')
    ax4.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé: {save_path}")

    # Calcul du speedup (CORRIGÉ)
    print(f"\n{'='*70}")
    print(f"RÉSULTATS DE L\'EXPÉRIENCE")
    print(f"{'='*70}")

def plot_vit_accuracy(csv_file, save_path='vit_accuracy_evolution.png'):
    """Plot ViT training and validation accuracy over epochs."""
    df = pd.read_csv(csv_file)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Accuracy Evolution
    ax1 = axes[0]
    ax1.plot(df['epoch'], df['train_acc'], 
             label='Training Accuracy', linewidth=2.5, marker='o', markersize=6, color='#2ca02c')
    ax1.plot(df['epoch'], df['val_acc'],
             label='Validation Accuracy', linewidth=2.5, marker='s', markersize=6, color='#d62728')
    ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('ViT Accuracy Evolution\n(Training vs Validation)', 
                  fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12, loc='lower right')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 100)

    # 2. Loss Evolution
    ax2 = axes[1]
    ax2.plot(df['epoch'], df['train_loss'],
             label='Training Loss', linewidth=2.5, marker='o', markersize=6, color='#1f77b4')
    ax2.plot(df['epoch'], df['val_loss'],
             label='Validation Loss', linewidth=2.5, marker='s', markersize=6, color='#ff7f0e')
    ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax2.set_title('ViT Loss Evolution', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=12, loc='upper right')
    ax2.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé: {save_path}")

    # Afficher les résultats
    print(f"\n{'='*70}")
    print(f"RÉSULTATS ViT TRAINING")
    print(f"{'='*70}")
    print(f"Epoch finale:             {df['epoch'].iloc[-1]}")
    print(f"Training Loss finale:     {df['train_loss'].iloc[-1]:.4f}")
    print(f"Validation Loss finale:   {df['val_loss'].iloc[-1]:.4f}")
    print(f"Training Accuracy:        {df['train_acc'].iloc[-1]:.2f}%")
    print(f"Validation Accuracy:      {df['val_acc'].iloc[-1]:.2f}%")
    print(f"Best Val Accuracy:        {df['val_acc'].max():.2f}% @ epoch {df.loc[df['val_acc'].idxmax(), 'epoch']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import sys
    
    print("\n=== MENU DE PLOTTING ===")
    print("1. Plot convergence comparison (2 CSV: standard + symmetric)")
    print("2. Plot convergence one (1 CSV: standard only)")
    print("3. Plot ViT accuracy (1 CSV: ViT metrics)")
    print("========================\n")
    
    choice = input("Choisir une option (1/2/3): ").strip()
    
    if choice == "1":
        csv_std = input("CSV Standard (défaut: metrics_standard.csv): ").strip() or "metrics_standard.csv"
        csv_sym = input("CSV Symmetric (défaut: metrics_symmetric.csv): ").strip() or "metrics_symmetric.csv"
        save_path = input("Nom de sortie (défaut: convergence_comparison.png): ").strip() or "convergence_comparison.png"
        plot_convergence_comparison(csv_std, csv_sym, save_path)
    
    elif choice == "2":
        csv_std = input("CSV Standard (défaut: metrics_standard.csv): ").strip() or "metrics_standard.csv"
        save_path = input("Nom de sortie (défaut: result_1M_100K_standard.png): ").strip() or "result_1M_100K_standard.png"
        plot_convergence_one(csv_std, save_path)
    
    elif choice == "3":
        csv_vit = input("CSV ViT (défaut: metrics_vit_standard.csv): ").strip() or "metrics_vit_standard.csv"
        save_path = input("Nom de sortie (défaut: vit_accuracy_evolution.png): ").strip() or "vit_accuracy_evolution.png"
        plot_vit_accuracy(csv_vit, save_path)

    
    else:
        print("Option invalide!")
        sys.exit(1)
