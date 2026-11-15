import numpy as np
from skfem import MeshLine, Basis, ElementLineP1, BilinearForm, solve, condense
from skfem.helpers import dot, grad

alpha = 2 / 3  # u(x) = x^alpha - x

def u_ex(x):
    return x**alpha - x

def F(x):  # primitive de u'(x)^2
    return (4 / 3) * x ** (1 / 3) - 2 * x ** (2 / 3) + x

@BilinearForm
def a(u, v, w):
    return dot(grad(u), grad(v))

def h1_error(nodes):
    basis = Basis(MeshLine(nodes), ElementLineP1())
    A = a.assemble(basis)
    # second membre exact : b_i = int u' phi_i' (f est singulière en 0)
    s = np.diff(u_ex(nodes)) / np.diff(nodes)
    b = np.zeros(len(nodes))
    b[1:-1] = s[:-1] - s[1:]
    uh = solve(*condense(A, b, D=basis.get_dofs()))
    # erreur H1 exacte, élément par élément
    h = np.diff(nodes)
    sh = np.diff(uh) / h
    du = np.diff(u_ex(nodes))
    e2 = np.sum(F(nodes[1:]) - F(nodes[:-1]) - 2 * sh * du + sh**2 * h)
    return np.sqrt(e2)

N = 5
x = np.linspace(0, 1, N + 1)
print("uniforme :", h1_error(x))      # attendu ≈ 0.44
print("resserré :", h1_error(x**3))   # attendu ≈ 0.29
