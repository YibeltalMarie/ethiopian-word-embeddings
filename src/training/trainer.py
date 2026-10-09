"""Training: single-pair step, dataset loss, epochs, and the full loop."""
import random

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


def evaluate_loss(E, U, pairs):
    """Dataset loss J = (1/N) * sum of one-pair losses, at FIXED weights.

    Only reads E and U: no gradients, no updates.

    Inputs:  E (V x d), U (d x V), pairs (list of (center ID, target ID))
    Output:  float, the mean loss over all pairs
    Raises:  ValueError if pairs is empty, IndexError if an ID is out of range
    """
    if not pairs:
        raise ValueError("cannot evaluate loss on an empty pair list")
    total = 0.0
    for i, t in pairs:
        _, s = forward(E, U, i)
        m, a, _ = compute_probabilities(s)
        total += compute_loss(s, t, m, a)
    return total / len(pairs)


def train_one_epoch(E, U, pairs, learning_rate, rng):
    """One epoch: visit every pair once, in a freshly shuffled order.

    The caller's pair list is never modified (a shuffled COPY is used).
    E and U are changed in place.

    Inputs:  E, U, pairs, learning_rate, rng (a random.Random instance)
    Output:  the shuffled order that was used (for logging and tests)
    """
    order = list(pairs)
    rng.shuffle(order)
    for i, t in order:
        train_pair(E, U, i, t, learning_rate)
    return order


def train(E, U, pairs, learning_rate, epochs, shuffle_seed, verbose=True):
    """Train for a number of epochs and record J before and after each one.

    J is always measured with evaluate_loss on the ORIGINAL pair list at frozen
    weights, never averaged from the losses seen while the weights were moving.

    Inputs:  E, U (changed in place), pairs, learning_rate (> 0),
             epochs (int >= 1), shuffle_seed (int, record it with the run),
             verbose (print J before training and after each epoch)
    Output:  history, a list of epochs + 1 values:
             [J before training, J after epoch 1, ..., J after the last epoch]
    Raises:  ValueError for empty pairs, learning_rate <= 0, or epochs < 1
    """
    if not pairs:
        raise ValueError("cannot train on an empty pair list")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError(f"epochs must be an integer >= 1, got {epochs!r}")
    rng = random.Random(shuffle_seed)
    history = [evaluate_loss(E, U, pairs)]
    if verbose:
        print(f"epoch 0 (before training): J = {history[0]:.6f}")
    for epoch in range(1, epochs + 1):
        train_one_epoch(E, U, pairs, learning_rate, rng)
        history.append(evaluate_loss(E, U, pairs))
        if verbose:
            print(f"epoch {epoch}: J = {history[-1]:.6f}")
    return history
