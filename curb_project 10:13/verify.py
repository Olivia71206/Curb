import numpy as np, time, sys
import curb_original as O, curb_fast as F
def norm(L): return sorted((tuple(sorted(a)),tuple(sorted(b))) for a,b in L)
rng=np.random.default_rng(1)
bad=0; tot=0; to=tf=0
cases=[(2,2,200),(3,3,150),(3,4,80),(4,3,80),(4,4,40),(5,5,15)]
for n1,n2,N in cases:
  for kind in ["unif","int"]:
    for _ in range(N):
      if kind=="unif": U1=rng.random((n1,n2));U2=rng.random((n1,n2))
      else: U1=rng.integers(0,4,(n1,n2)).astype(float);U2=rng.integers(0,4,(n1,n2)).astype(float)
      t=time.time(); a=norm(O.find_all_minimal_curb_sets(U1,U2)); to+=time.time()-t
      t=time.time(); b=norm(F.find_all_minimal_curb_sets(U1,U2,use_iesds=False)); c=norm(F.find_all_minimal_curb_sets(U1,U2,use_iesds=True)); tf+=time.time()-t
      tot+=1
      if not (a==b==c): bad+=1; print("MISMATCH",kind,n1,n2,a,b,c) if bad<5 else None
print("games",tot,"mismatches",bad,"orig time %.1fs fast(both) %.1fs"%(to,tf))
