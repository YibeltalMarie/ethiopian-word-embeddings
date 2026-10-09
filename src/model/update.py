"""SGD parameter update."""


def update_parameters(E, U, i, grad_U, grad_h, learning_rate):
    """SGD update, changing E and U IN PLACE.

        U    <- U    - learning_rate * grad_U
        E[i] <- E[i] - learning_rate * grad_h

    Only row i of E is touched; every other row of E is left unchanged.
    The gradients must already have been computed from the old E and U.

    Inputs:  E (V x d), U (d x V), i (center ID), grad_U (d x V),
             grad_h (length d), learning_rate (float)
    Output:  None (E and U are modified directly)
    Raises:  IndexError if i is out of range, ValueError on shape mismatch
    """
    if not 0 <= i < len(E):
        raise IndexError(f"center ID {i} out of range for V={len(E)}")
    d = len(U)
    V = len(U[0]) if d else 0
    if len(grad_U) != d or any(len(row) != V for row in grad_U):
        raise ValueError("grad_U shape does not match U")
    if len(grad_h) != len(E[i]) or len(E[i]) != d:
        raise ValueError("grad_h shape does not match E[i] and U")
    for k in range(d):
        for j in range(V):
            U[k][j] -= learning_rate * grad_U[k][j]
    for k in range(d):
        E[i][k] -= learning_rate * grad_h[k]
