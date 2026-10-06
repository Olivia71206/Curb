import numpy as np
from scipy.optimize import linprog
from itertools import combinations


def all_nonempty_subsets(n=4):
    """
    Generate all non-empty subsets of {0, 1, ..., n-1}.
    """
    subsets = []

    for r in range(1, n + 1):
        for combo in combinations(range(n), r):
            subsets.append(list(combo))

    return subsets


def can_be_best_response_p1(target_strategy, X2, U1):
    """
    Check whether Player 1's target_strategy can be a best response
    to some belief supported on Player 2's strategy subset X2.
    """

    # We only care about feasibility, so the objective is zero
    c = np.zeros(len(X2))

    # Belief probabilities must sum to 1
    A_eq = [np.ones(len(X2))]
    b_eq = [1]

    # Each belief probability must be between 0 and 1
    bounds = [(0, 1)] * len(X2)

    # Best-response constraints
    A_ub = []
    b_ub = []

    for other_strategy in range(4):

        if other_strategy == target_strategy:
            continue

        # EU(other) - EU(target) <= 0
        difference = (
            U1[other_strategy, X2]
            - U1[target_strategy, X2]
        )

        A_ub.append(difference)
        b_ub.append(0)

    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs"
    )

    return result.success


def can_be_best_response_p2(target_strategy, X1, U2):
    """
    Check whether Player 2's target_strategy can be a best response
    to some belief supported on Player 1's strategy subset X1.
    """

    c = np.zeros(len(X1))

    # Belief probabilities must sum to 1
    A_eq = [np.ones(len(X1))]
    b_eq = [1]

    bounds = [(0, 1)] * len(X1)

    A_ub = []
    b_ub = []

    for other_strategy in range(4):

        if other_strategy == target_strategy:
            continue

        # Player 2 chooses columns
        # EU(other) - EU(target) <= 0
        difference = (
            U2[X1, other_strategy]
            - U2[X1, target_strategy]
        )

        A_ub.append(difference)
        b_ub.append(0)

    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs"
    )

    return result.success


def is_curb(X1, X2, U1, U2):
    """
    Check whether X1 x X2 is a CURB set.
    """

    # Player 1 strategies outside X1
    outside_p1 = [s for s in range(4) if s not in X1]

    for s in outside_p1:
        if can_be_best_response_p1(s, X2, U1):
            return False

    # Player 2 strategies outside X2
    outside_p2 = [s for s in range(4) if s not in X2]

    for s in outside_p2:
        if can_be_best_response_p2(s, X1, U2):
            return False

    return True


def find_all_curb_sets(U1, U2):
    """
    Find all CURB sets in a 4x4 game.
    """

    all_subsets = all_nonempty_subsets(4)

    curb_sets = []

    for X1 in all_subsets:
        for X2 in all_subsets:
            if is_curb(X1, X2, U1, U2):
                curb_sets.append((X1, X2))

    return curb_sets


def find_minimal_curb_sets(U1, U2):
    """
    Find all minimal CURB sets in a 4x4 game.
    """

    curb_sets = find_all_curb_sets(U1, U2)

    minimal_curb_sets = []

    for X1, X2 in curb_sets:

        is_minimal = True

        for Y1, Y2 in curb_sets:

            if (Y1, Y2) == (X1, X2):
                continue

            if (
                set(Y1).issubset(set(X1))
                and set(Y2).issubset(set(X2))
            ):
                is_minimal = False
                break

        if is_minimal:
            minimal_curb_sets.append((X1, X2))

    return minimal_curb_sets