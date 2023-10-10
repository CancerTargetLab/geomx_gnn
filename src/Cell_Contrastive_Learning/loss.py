import torch
import torch.nn.functional as F

def add_contrastive_loss(hidden,
                         hidden_norm=True,
                         temperature=0.5,
                         weights=1.0):
    """
    Compute contrastive loss for a model.
    Implemented following example from A Simple Framework for Contrastive Learning of Visual Representations,
    https://github.com/google-research/simclr

    Args:
        hidden: Hidden vector (`Tensor`) of shape (bsz, dim).
        hidden_norm: Whether or not to use normalization on the hidden vector.
        temperature: A floating-point number for temperature scaling.
        weights: A weighting number or vector. :TODO: make useable if ever needed

    Returns:
        A loss scalar.
        The logits for the contrastive prediction task.
        The labels for the contrastive prediction task.
    """

    LARGE_NUM = 1e9
    
    # Get (normalized) hidden1 and hidden2.
    if hidden_norm:
        hidden = F.normalize(hidden, p=2, dim=-1)
    hidden1, hidden2 = torch.split(hidden, split_size_or_sections=hidden.size(0) // 2, dim=0)
    batch_size = hidden1.size(0)
    # Create labels ad mask for contrastive prediction task.
    labels = F.one_hot(torch.arange(batch_size*2)%32, batch_size).float().to(hidden.device)
    masks = F.one_hot(torch.arange(batch_size), batch_size).to(hidden.device)

    # Compute cosine similarities (dot products) between hidden vectors.
    logits_aa = torch.mm(hidden1, hidden1.t()) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = torch.mm(hidden2, hidden2.t()) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = torch.mm(hidden1, hidden2.t()) / temperature
    logits_ba = torch.mm(hidden2, hidden1.t()) / temperature
    logits = torch.cat((logits_ab, logits_ba), dim=0)

    # Calculate the contrastive loss using the InfoNCE (Noise Contrastive Estimation) loss.
    loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=0), labels)
    loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=0), labels)
    loss = loss_a + loss_b

    return loss, logits, labels
