"""Training step for one (center, target) pair."""
from src.model.forward import forward, compute_probabilities
from src.model.loss import compute_loss
from src.model.gradients import compute_gradients
from src.model.update import update_parameters


def train_pair(E, U, i, t, learning_rate):
    """Train on one (center i, target t) pair and return that pair's loss.

    Follows the required order:
        1-2. read old E[i] -> h
        3.   scores
        4.   probabilities
        5.   loss
        6-7. both gradients, from the OLD h and OLD U
        8-9. update U, then update only E[i]

    The returned loss is the one measured BEFORE the update.
    """
    h, s = forward(E, U, i)
    m, a, p = compute_probabilities(s)
    loss = compute_loss(s, t, m, a)
    grad_U, grad_h = compute_gradients(h, U, p, t)
    update_parameters(E, U, i, grad_U, grad_h, learning_rate)
    return loss
