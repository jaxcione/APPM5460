import numpy as np
import sympy as sp

t=sp.symbols('t',real=True)

A=sp.Matrix([[1,t],[0,-1]])
B=sp.Matrix([[t,t**2/2],[0,-t]])
eigen_data=B.eigenvects()
P=sp.Matrix([[-t/4,1],[1,0]])
P_inv=P.inv()
lam=sp.Matrix([[-t,0],[0,t]])

P_deriv = sp.diff(P, t)
P_inv_deriv = sp.diff(P_inv, t)
lam_deriv = sp.diff(lam, t)
e_lam = sp.diag(sp.exp(-t), sp.exp(t))

print(P@lam@P_inv,'B decomp') #just checking it worked 

print(P_inv)

for val, mult, vects in eigen_data:
    print(f"\nEigenvalue: {val}")
    print(f"Multiplicity: {mult}")
    print("Eigenvector(s):")
    for v in vects:
        sp.pprint(v)

LHS = (P_deriv@e_lam@P_inv)+(P@lam_deriv@e_lam@P_inv)+(P@e_lam@P_inv_deriv)
RHS=A@P@e_lam@P_inv

print(LHS,"Left hand side")
print(RHS,'Right hand side')

print("Net difference:", LHS-RHS)