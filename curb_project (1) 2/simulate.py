"""
Monte Carlo simulation of minimal CURB sets in random games.
Payoffs ~ i.i.d. Uniform[0,1] for both players.

Usage: python simulate.py N_PER_SIZE OUT.csv
Each row = one game.
"""
import sys, csv, json, time
import numpy as np
from multiprocessing import Pool
from curb_fast import iesds, CurbSolver

SIZES = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]


def pure_nash(U1, U2):
    br1 = U1 >= U1.max(axis=0, keepdims=True)
    br2 = U2 >= U2.max(axis=1, keepdims=True)
    return int((br1 & br2).sum())


def one_game(args):
    n1, n2, seed = args
    rng = np.random.default_rng(seed)
    U1, U2 = rng.random((n1, n2)), rng.random((n1, n2))
    S1, S2 = iesds(U1, U2)
    curbs = CurbSolver(U1, U2, S1, S2).minimal_curb_sets()
    sizes = sorted((len(a), len(b)) for a, b in curbs)
    return {
        "n1": n1, "n2": n2, "seed": seed,
        "iesds1": len(S1), "iesds2": len(S2),
        "n_curb": len(curbs),
        "curb_sizes": json.dumps(sizes),
        "n_pure_ne": pure_nash(U1, U2),
    }


if __name__ == "__main__":
    N = int(sys.argv[1]); out = sys.argv[2]
    jobs = [(n1, n2, 1_000_000 * k + i)
            for k, (n1, n2) in enumerate(SIZES) for i in range(N)]
    t = time.time()
    with Pool() as p, open(out, "w", newline="") as f:
        w = None
        for j, row in enumerate(p.imap(one_game, jobs, chunksize=50)):
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader()
            w.writerow(row)
            if j % 2000 == 0:
                print(j, len(jobs), round(time.time() - t), flush=True)
    print("done", round(time.time() - t), flush=True)
