import torch
import torch.nn.functional as F

def add_contrastive_loss(hidden,
                         hidden_norm=True,
                         temperature=1.0,
                         weights=1.0):
    """
    Compute contrastive loss for a model.
    Implemented following example from A Simple Framework for Contrastive Learning of Visual Representations,
    https://github.com/google-research/simclr

    Args:
        hidden: Hidden vector (`Tensor`) of shape (bsz, dim).
        hidden_norm: Whether or not to use normalization on the hidden vector.
        temperature: A floating-point number for temperature scaling.
        weights: A weighting number or vector.

    Returns:
        A loss scalar.
        The logits for the contrastive prediction task.
        The labels for the contrastive prediction task.
    """
    
    # Get (normalized) hidden1 and hidden2.
    if hidden_norm:
        hidden = F.normalize(hidden, p=2, dim=-1)
    hidden1, hidden2 = torch.split(hidden, split_size_or_sections=hidden.size(0) // 2, dim=0)
    batch_size = hidden1.size(0)

    # Create labels for contrastive prediction task.
    labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)]).to(hidden.device)

    # Compute cosine similarities (dot products) between hidden vectors.
    logits_aa = torch.mm(hidden1, hidden1.t()) / temperature
    logits_bb = torch.mm(hidden2, hidden2.t()) / temperature
    logits_ab = torch.mm(hidden1, hidden2.t()) / temperature
    logits_ba = torch.mm(hidden2, hidden1.t()) / temperature

    # Calculate the contrastive loss using the InfoNCE (Noise Contrastive Estimation) loss.
    loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels, weight=weights)
    loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels, weight=weights)
    loss = loss_a + loss_b

    return loss, logits_ab, labels
