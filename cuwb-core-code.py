import numpy as np
from scipy.optimize import linprog


def all_better_replies_p1(S1, X1, X2, U1):
    """
    Finds all pure strategies in S1 that yield at least the payoff of
    AT LEAST ONE reference strategy in X1 against some belief over X2.
    """
    if not X2 or not X1:
        return set()

    X2_list = sorted(list(X2))
    num_vars = len(X2_list)
    better_replies = set()

    for r in S1:
        # Strategies already in X1 trivially beat/tie themselves.
        if r in X1:
            better_replies.add(r)
            continue

        # Check if candidate r beats or ties AT LEAST ONE reference
        # strategy r_hat in X1.
        for r_hat in X1:
            # Single inequality constraint: EU(r_hat) - EU(r) <= 0.
            diff = [U1[r_hat, c] - U1[r, c] for c in X2_list]
            A_ub = [diff]
            b_ub = [0.0]

            c_obj = np.zeros(num_vars)
            A_eq = [np.ones(num_vars)]
            b_eq = [1.0]
            bounds = [(0, 1)] * num_vars

            res = linprog(
                c_obj,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )

            if res.success:
                better_replies.add(r)
                break  # Stop checking other reference strategies for r.

    return better_replies


def all_better_replies_p2(S2, X2, X1, U2):
    """
    Finds all pure strategies in S2 that yield at least the payoff of
    AT LEAST ONE reference strategy in X2 against some belief over X1.
    """
    if not X1 or not X2:
        return set()

    X1_list = sorted(list(X1))
    num_vars = len(X1_list)
    better_replies = set()

    for c in S2:
        # Strategies already in X2 trivially beat/tie themselves.
        if c in X2:
            better_replies.add(c)
            continue

        # Check if candidate c beats or ties AT LEAST ONE reference
        # strategy c_hat in X2.
        for c_hat in X2:
            # Single inequality constraint: EU(c_hat) - EU(c) <= 0.
            diff = [U2[r, c_hat] - U2[r, c] for r in X1_list]
            A_ub = [diff]
            b_ub = [0.0]

            c_obj = np.zeros(num_vars)
            A_eq = [np.ones(num_vars)]
            b_eq = [1.0]
            bounds = [(0, 1)] * num_vars

            res = linprog(
                c_obj,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )

            if res.success:
                better_replies.add(c)
                break  # Stop checking other reference strategies for c.

    return better_replies


def min_containing_cuwb(seed_strategy, player, U1, U2):
    """
    Computes the smallest CUWB set containing a given root/seed strategy.
    """
    num_p1, num_p2 = U1.shape
    all_S1 = set(range(num_p1))
    all_S2 = set(range(num_p2))

    if player == 1:
        X1 = {seed_strategy}
        X2 = set()
    else:
        X1 = set()
        X2 = {seed_strategy}

    converged = False

    while not converged:
        converged = True

        # Pass X2 (reference set) and X1 (opponent belief set).
        new_X2 = all_better_replies_p2(all_S2, X2, X1, U2)

        if not new_X2.issubset(X2):
            converged = False
            X2 = X2.union(new_X2)

        # Pass X1 (reference set) and X2 (opponent belief set).
        new_X1 = all_better_replies_p1(all_S1, X1, X2, U1)

        if not new_X1.issubset(X1):
            converged = False
            X1 = X1.union(new_X1)

    return X1, X2


def find_all_minimal_cuwb_sets(U1, U2):
    """
    Finds all minimal CUWB sets using min_containing_cuwb
    as a building block.
    """
    num_p1, num_p2 = U1.shape
    candidate_cuwbs = []

    for r in range(num_p1):
        cuwb = min_containing_cuwb(r, player=1, U1=U1, U2=U2)

        if cuwb not in candidate_cuwbs:
            candidate_cuwbs.append(cuwb)

    for c in range(num_p2):
        cuwb = min_containing_cuwb(c, player=2, U1=U1, U2=U2)

        if cuwb not in candidate_cuwbs:
            candidate_cuwbs.append(cuwb)

    minimal_cuwb_sets = []

    for X1, X2 in candidate_cuwbs:
        is_minimal = True

        for Y1, Y2 in candidate_cuwbs:
            if (Y1, Y2) == (X1, X2):
                continue

            if Y1.issubset(X1) and Y2.issubset(X2):
                is_minimal = False
                break

        if is_minimal and (X1, X2) not in minimal_cuwb_sets:
            minimal_cuwb_sets.append((X1, X2))

    return minimal_cuwb_sets
