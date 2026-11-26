#!/usr/bin/env python3
"""Which triples (f, g, h) of bijections keep c = a op b on Z_p, that is h(a op b) = f(a) op g(b)? Brute force for addition and
midpoint (docs/READING_LIST_arithmetic_symmetry.md). A triple is an autotopism, and an automorphism when f = g = h.
    python3 analysis/autotopism_check.py
"""
import itertools
def solve(p, op):
    table={(a,b):op(a,b) for a in range(p) for b in range(p)}
    inv={ (a,c):b for (a,b),c in table.items() }              # a op b = c -> b
    invl={ (b,c):a for (a,b),c in table.items() }             # a op b = c -> a
    out=[]
    for f in itertools.permutations(range(p)):
        for g0 in range(p):
            h=[None]*p
            for a in range(p): h[table[(a,0)]]=table[(f[a],g0)]
            if None in h or len(set(h))<p: continue
            g=[inv[(f[0],h[table[(0,y)]])] for y in range(p)]
            if len(set(g))<p: continue
            if all(h[table[(a,b)]]==table[(f[a],g[b])] for a in range(p) for b in range(p)):
                out.append((f,tuple(g),tuple(h)))
    return out
for p in (5,7):
    half=(p+1)//2
    add=lambda a,b:(a+b)%p
    mid=lambda a,b:(a+b)*half%p                                # (a + b) / 2 mod p
    A=solve(p,add); M=solve(p,mid)
    aut=lambda L:[t for t in L if t[0]==t[1]==t[2]]
    trans=lambda L,tr:[s for s in range(p) if ((tuple((x+s)%p for x in range(p)),)*3 in L)]
    print(f"Z_{p}: addition autotopisms {len(A)} (p^2(p-1) = {p*p*(p-1)}), same map on all three {len(aut(A))} (p-1 = {p-1}), uniform shifts kept {trans(set(A),0)}")
    print(f"Z_{p}: midpoint autotopisms {len(M)}, same map on all three {len(aut(M))} (p(p-1) = {p*(p-1)}), uniform shifts kept {trans(set(M),0)}")
    # every autotopism of addition has the published form f=t(x+u), g=t(x+v), h=t(x+u+v)
    forms={(tuple(t*(x+u)%p for x in range(p)),tuple(t*(x+v)%p for x in range(p)),tuple(t*(x+u+v)%p for x in range(p))) for t in range(1,p) for u in range(p) for v in range(p)}
    print("  published form equals the brute-force set:",forms==set(A))
    # midpoint is addition with its rows and columns relabelled x -> x/2 (an isotopy)
    print("  midpoint = addition relabelled:",all(mid(a,b)==add(a*half%p,b*half%p) for a in range(p) for b in range(p)))
