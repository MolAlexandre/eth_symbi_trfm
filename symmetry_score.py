import torch
import torch.nn as nn
import numpy as np

def symmetry_score(W_qk: torch.Tensor) -> float:
    """
    Calcule le score de symétrie d'une matrice.

    Score de symétrie: s = (||M_s||^2 - ||M_n||^2) / ||M||^2
    où M_s est la partie symétrique et M_n la partie antisymétrique.

    Args:
        W_qk: Matrice Wqk [d_model, d_model]

    Returns:
        float: Score de symétrie entre -1 (antisymétrique) et 1 (symétrique)
    """
    # Décomposition en parties symétrique et antisymétrique
    M_s = 0.5 * (W_qk + W_qk.T)  # Partie symétrique
    M_n = 0.5 * (W_qk - W_qk.T)  # Partie antisymétrique

    # Calcul des normes de Frobenius au carré
    norm_M_s_sq = torch.norm(M_s, p='fro').pow(2)
    norm_M_n_sq = torch.norm(M_n, p='fro').pow(2)
    norm_M_sq = norm_M_s_sq + norm_M_n_sq

    # Score de symétrie
    score = (norm_M_s_sq - norm_M_n_sq) / norm_M_sq
    return score.item()


def compute_model_symmetry(model):
    """
    Calcule les scores de symétrie pour toutes les couches d'attention.
    Compatible avec BERT (BERTForMLM) et ViT (VITForClassification).

    Args:
        model: Instance de BERTForMLM ou VITForClassification

    Returns:
        dict: Scores par couche + moyenne
    """
    symmetry_scores = {}

    # Détection automatique de l'architecture
    if hasattr(model, 'encoder'):
        # Architecture BERT: model.encoder.layers
        layers = model.encoder.layers
    elif hasattr(model, 'encoder_layers'):
        # Architecture ViT: model.encoder_layers
        layers = model.encoder_layers
    else:
        raise AttributeError(
            f"Model type {type(model).__name__} not supported. "
            "Expected 'encoder.layers' (BERT) or 'encoder_layers' (ViT)."
        )

    with torch.no_grad():
        for layer_idx, layer in enumerate(layers):
            # Récupérer Wq et Wk
            Wq = layer.attention.query.weight  # (d_model, d_model)
            Wk = layer.attention.key.weight    # (d_model, d_model)

            # Calculer Wqk = Wq @ Wk.T
            Wqk = Wq @ Wk.T

            # Score de symétrie
            score = symmetry_score(Wqk)
            symmetry_scores[f'layer_{layer_idx}'] = score

    # Moyenne sur toutes les couches
    avg_score = np.mean(list(symmetry_scores.values()))
    symmetry_scores['average'] = avg_score

    return symmetry_scores


def log_symmetry_scores(symmetry_scores, epoch, prefix=""):
    """
    Affiche les scores de symétrie de manière formatée.
    """
    print(f"\n{prefix}[Epoch {epoch+1}] Symmetry Scores:")
    print("-" * 50)
    for layer_name, score in symmetry_scores.items():
        if layer_name != 'average':
            print(f"  {layer_name}: {score:+.4f}")
    print("-" * 50)
    print(f"  Average: {symmetry_scores['average']:+.4f}")
    print("-" * 50)
