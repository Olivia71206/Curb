"""
Fast CURB computation for 2-player normal-form games.

Same definition as Basu & Weibull (1991) and as curb_core_code_2.py:
a strategy is added if it is a best reply to SOME belief over the
opponent's current set (weak inequality, ties count).

Speed-ups vs. the original:
  1. Pure-belief prefilter: if r is a best reply to some pure c in X_opp,
     it is certainly in the set (no LP needed).
  2. Best-reply sets are cached per (player, frozenset(X_opp)), so
     different seeds that reach the same intermediate sets reuse work.
  3. Optional IESDS (iterated elimination of strictly dominated
     strategies, domination by mixed strategies) before the search.
"""
import numpy as np
from scipy.optimize import linprog

TOL = 1e-9


# ----------------------------------------------------------------------
# Best replies
# ----------------------------------------------------------------------
def _is_best_reply_to_some_belief(U, r, own, opp):
    """
    U[own_strategy, opp_strategy] = payoff of the player choosing rows.
    Is row r a best reply to some belief p over columns `opp`,
    when compared against all rows in `own`?
    """
    opp = list(opp)
    others = [k for k in own if k != r]
    if not others:
        return True
    # Pure-belief prefilter (vertices of the simplex).
    sub = U[np.ix_(list(own), opp)]
    col_max = sub.max(axis=0)
    if np.any(U[r, opp] >= col_max - TOL):
        return True
    # LP feasibility: sum_c p_c (U[k,c]-U[r,c]) <= 0 for all k != r.
    A_ub = U[np.ix_(others, opp)] - U[r, opp]
    n = len(opp)
    res = linprog(
        np.zeros(n), A_ub=A_ub, b_ub=np.zeros(len(others)),
        A_eq=np.ones((1, n)), b_eq=[1.0], bounds=[(0, None)] * n,
        method="highs",
    )
    return res.status == 0


class CurbSolver:
    """Holds a game and caches best-reply sets."""

    def __init__(self, U1, U2, S1=None, S2=None):
        # Internally both players are "row" players:
        #   P[0] = U1   (rows = P1 strategies, cols = P2 strategies)
        #   P[1] = U2.T (rows = P2 strategies, cols = P1 strategies)
        self.P = [np.asarray(U1, float), np.asarray(U2, float).T]
        self.S = [
            frozenset(S1) if S1 is not None else frozenset(range(U1.shape[0])),
            frozenset(S2) if S2 is not None else frozenset(range(U1.shape[1])),
        ]
        self.cache = [{}, {}]

    def best_replies(self, i, X_opp):
        """All strategies of player i (0 or 1) that are best replies
        to some belief over X_opp."""
        if not X_opp:
            return frozenset()
        key = frozenset(X_opp)
        c = self.cache[i]
        if key not in c:
            own = sorted(self.S[i])
            c[key] = frozenset(
                r for r in own
                if _is_best_reply_to_some_belief(self.P[i], r, own, sorted(key))
            )
        return c[key]

    def closure(self, seed, player):
        """Smallest CURB set containing `seed` (player 1 or 2)."""
        X = [set(), set()]
        X[player - 1].add(seed)
        while True:
            changed = False
            # same order as original: expand P2, then P1
            for i in (1, 0):
                new = self.best_replies(i, X[1 - i])
                if not new <= X[i]:
                    X[i] |= new
                    changed = True
            if not changed:
                return frozenset(X[0]), frozenset(X[1])

    def minimal_curb_sets(self):
        cands = set()
        for r in self.S[0]:
            cands.add(self.closure(r, 1))
        for c in self.S[1]:
            cands.add(self.closure(c, 2))
        mins = [
            X for X in cands
            if not any(Y != X and Y[0] <= X[0] and Y[1] <= X[1] for Y in cands)
        ]
        return sorted(mins, key=lambda X: (sorted(X[0]), sorted(X[1])))


# ----------------------------------------------------------------------
# IESDS (domination by mixed strategies)
# ----------------------------------------------------------------------
def _strictly_dominated(U, r, own, opp):
    """Is row r strictly dominated by a mixture of the other rows in
    `own`, against columns `opp`?  max eps s.t. q.U[:,c] >= U[r,c]+eps."""
    others = [k for k in own if k != r]
    if not others:
        return False
    opp = list(opp)
    m = len(others)
    # variables: q_1..q_m, eps ; minimise -eps
    c_obj = np.zeros(m + 1)
    c_obj[-1] = -1.0
    # -(q.U[:,c]) + eps <= -U[r,c]
    A_ub = np.hstack([-U[np.ix_(others, opp)].T, np.ones((len(opp), 1))])
    b_ub = -U[r, opp]
    A_eq = np.hstack([np.ones((1, m)), np.zeros((1, 1))])
    res = linprog(
        c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
        bounds=[(0, None)] * m + [(None, 1e6)], method="highs",
    )
    return res.status == 0 and -res.fun > TOL


def iesds(U1, U2):
    """Returns surviving strategy sets (S1, S2) after iterated
    elimination of strictly dominated strategies (mixed domination)."""
    P = [np.asarray(U1, float), np.asarray(U2, float).T]
    S = [list(range(U1.shape[0])), list(range(U1.shape[1]))]
    changed = True
    while changed:
        changed = False
        for i in (0, 1):
            for r in list(S[i]):
                if len(S[i]) > 1 and _strictly_dominated(P[i], r, S[i], S[1 - i]):
                    S[i].remove(r)
                    changed = True
    return set(S[0]), set(S[1])


# ----------------------------------------------------------------------
# Convenience wrappers
# ----------------------------------------------------------------------
def find_all_minimal_curb_sets(U1, U2, use_iesds=True):
    """Drop-in replacement for the original function.
    Returns list of (X1, X2) as python sets, in ORIGINAL indices."""
    if use_iesds:
        S1, S2 = iesds(U1, U2)
    else:
        S1 = S2 = None
    solver = CurbSolver(U1, U2, S1, S2)
    return [(set(a), set(b)) for a, b in solver.minimal_curb_sets()]
