import numpy as np
from scipy.optimize import linprog


def all_conditionally_rational_p1(S1, X2, U1):
    """
    Finds all pure strategies in S1 that are best responses to at least
    one belief (probability distribution) over Player 2's strategy set X2.
    """
    if not X2:
        return set()

    X2_list = sorted(list(X2))
    num_vars = len(X2_list)
    best_responses = set()

    # Check LFP feasibility for each candidate strategy r in S1.
    for r in S1:
        A_ub = []
        b_ub = []

        for r_prime in S1:
            if r_prime == r:
                continue

            # EU(r') - EU(r) <= 0
            diff = [U1[r_prime, c] - U1[r, c] for c in X2_list]
            A_ub.append(diff)
            b_ub.append(0.0)

        # Objective is zero (pure feasibility problem).
        c_obj = np.zeros(num_vars)

        # Probability sum constraint: sum(p_c) = 1.
        A_eq = [np.ones(num_vars)]
        b_eq = [1.0]

        # Non-negativity bounds: p_c >= 0.
        bounds = [(0, 1)] * num_vars

        if len(A_ub) == 0:
            best_responses.add(r)
        else:
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
                best_responses.add(r)

    return best_responses


def all_conditionally_rational_p2(S2, X1, U2):
    """
    Finds all pure strategies in S2 that are best responses to at least
    one belief (probability distribution) over Player 1's strategy set X1.
    """
    if not X1:
        return set()

    X1_list = sorted(list(X1))
    num_vars = len(X1_list)
    best_responses = set()

    for c in S2:
        A_ub = []
        b_ub = []

        for c_prime in S2:
            if c_prime == c:
                continue

            # EU(c') - EU(c) <= 0
            diff = [U2[r, c_prime] - U2[r, c] for r in X1_list]
            A_ub.append(diff)
            b_ub.append(0.0)

        c_obj = np.zeros(num_vars)
        A_eq = [np.ones(num_vars)]
        b_eq = [1.0]
        bounds = [(0, 1)] * num_vars

        if len(A_ub) == 0:
            best_responses.add(c)
        else:
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
                best_responses.add(c)

    return best_responses


def min_containing_curb(seed_strategy, player, U1, U2):
    """
    Computes the smallest CURB set containing a given root/seed strategy.
    Runs in polynomial time O(n^2 * LFP(n)).
    """
    num_p1, num_p2 = U1.shape
    all_S1 = set(range(num_p1))
    all_S2 = set(range(num_p2))

    # Initialize with the root seed strategy.
    if player == 1:
        X1 = {seed_strategy}
        X2 = set()
    else:
        X1 = set()
        X2 = {seed_strategy}

    converged = False

    while not converged:
        converged = True

        # 1. Expand Player 2's set with best responses to
        #    Player 1's current set.
        new_X2 = all_conditionally_rational_p2(all_S2, X1, U2)

        if not new_X2.issubset(X2):
            converged = False

        # CRITICAL: Union new strategies with existing ones.
        X2 = X2.union(new_X2)

        # 2. Expand Player 1's set with best responses to
        #    Player 2's current set.
        new_X1 = all_conditionally_rational_p1(all_S1, X2, U1)

        if not new_X1.issubset(X1):
            converged = False

        # CRITICAL: Union new strategies with existing ones.
        X1 = X1.union(new_X1)

    return X1, X2


def find_all_minimal_curb_sets(U1, U2):
    """
    Finds all minimal CURB sets using min_containing_curb
    as a building block.
    """
    num_p1, num_p2 = U1.shape
    candidate_curbs = []

    # Run min_containing_curb using each strategy as a seed.
    for r in range(num_p1):
        curb = min_containing_curb(r, player=1, U1=U1, U2=U2)

        if curb not in candidate_curbs:
            candidate_curbs.append(curb)

    for c in range(num_p2):
        curb = min_containing_curb(c, player=2, U1=U1, U2=U2)

        if curb not in candidate_curbs:
            candidate_curbs.append(curb)

    # Filter candidates to retain only minimal CURB sets.
    minimal_curb_sets = []

    for X1, X2 in candidate_curbs:
        is_minimal = True

        for Y1, Y2 in candidate_curbs:
            if (Y1, Y2) == (X1, X2):
                continue

            # If Y is strictly contained within X, X is not minimal.
            if Y1.issubset(X1) and Y2.issubset(X2):
                is_minimal = False
                break

        if is_minimal and (X1, X2) not in minimal_curb_sets:
            minimal_curb_sets.append((X1, X2))

    return minimal_curb_sets
