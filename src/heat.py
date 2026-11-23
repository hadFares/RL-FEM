"""Équation de la chaleur stationnaire sur un assemblage de plaques minces.

Modèle (plaque mince : la température ne varie pas dans l'épaisseur) :
sur chaque plaque i, de conductivité k_i, d'épaisseur e_i et de source q_i,

    -div(k_i e_i grad T) + 2 h (T - T_inf) = q_i            dans la plaque i

  - k_i e_i grad T : flux conduit dans le plan, par unité de longueur [W/m]
  - 2 h (T - T_inf) : convection sur les deux faces (dessus et dessous) [W/m²]
  - q_i : puissance surfacique injectée [W/m²]

Conditions aux limites :
  - bord extérieur de l'assemblage : convection par la tranche,
        -k e dT/dn = h e (T - T_inf)
  - interface entre deux plaques (contact parfait) : T continue et flux k e dT/dn
    continu. Les deux conditions sont automatiques avec un maillage conforme
    (nœuds partagés) et la forme faible ci-dessous : rien à écrire.

Forme faible : trouver T dans H¹ tel que, pour toute fonction test v,

    ∫ k e ∇T·∇v + ∫ 2h T v + ∮_ext h e T v  =  ∫ (q + 2h T_inf) v + ∮_ext h e T_inf v

Aucune condition de Dirichlet n'est nécessaire : h > 0 rend le problème coercif.

Unités : le générateur travaille dans le carré [0, 100], et les épaisseurs sont
dans E_RANGE = (1, 5). On interprète les deux en millimètres (LENGTH_UNIT),
puis on convertit en mètres pour que k, h et q restent en unités SI.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from skfem import Basis, BilinearForm, ElementTriP0, ElementTriP1, FacetBasis, LinearForm, MeshTri
from skfem.helpers import dot, grad

from assembly import Assembly
from mesh import Mesh

LENGTH_UNIT = 1e-3  # 1 unité du générateur = 1 mm (coordonnées et épaisseurs)


@dataclass
class HeatProblem:
    """Système linéaire A T = b de l'équation de la chaleur discrétisée en P1.

    T[j] est la température au nœud j de `basis.mesh` (même ordre que mesh.nodes).
    """

    basis: Basis          # base P1 sur tout l'assemblage
    edge_basis: FacetBasis  # base P1 sur le bord extérieur (convection par la tranche)
    fields: dict[str, np.ndarray]  # ke, e, q par triangle (SI), voir piece_fields
    A: csr_matrix
    b: np.ndarray


# Formes bilinéaires et linéaires (w.ke, w.e, w.q : champs constants par triangle)
@BilinearForm
def _surface_lhs(u, v, w):
    return w.ke * dot(grad(u), grad(v)) + 2.0 * w.h * u * v


@LinearForm
def _surface_rhs(v, w):
    return (w.q + 2.0 * w.h * w.T_inf) * v


@BilinearForm
def _edge_lhs(u, v, w):
    return w.h * w.e * u * v


@LinearForm
def _edge_rhs(v, w):
    return w.h * w.e * w.T_inf * v


def to_skfem(mesh: Mesh) -> MeshTri:
    """Convertit le maillage Gmsh en MeshTri scikit-fem, en mètres."""
    return MeshTri(mesh.nodes.T * LENGTH_UNIT, mesh.triangles.T)


def piece_fields(assembly: Assembly, mesh: Mesh) -> dict[str, np.ndarray]:
    """Propriétés physiques de chaque triangle (SI), lues sur la plaque qui le contient."""
    k = np.array([p.k for p in assembly.plates])
    e = np.array([p.e for p in assembly.plates]) * LENGTH_UNIT
    q = np.array([p.q for p in assembly.plates])
    return {"ke": (k * e)[mesh.piece], "e": e[mesh.piece], "q": q[mesh.piece]}


def assemble_heat(assembly: Assembly, mesh: Mesh) -> HeatProblem:
    """Assemble la matrice et le second membre de l'équation de la chaleur."""
    m = to_skfem(mesh)
    fields = piece_fields(assembly, mesh)

    basis = Basis(m, ElementTriP1())
    p0 = basis.with_element(ElementTriP0())  # pour évaluer les champs constants par triangle
    surf = {name: p0.interpolate(val) for name, val in fields.items()}
    A = _surface_lhs.assemble(basis, ke=surf["ke"], h=assembly.h)
    b = _surface_rhs.assemble(basis, q=surf["q"], h=assembly.h, T_inf=assembly.T_inf)

    # Bord extérieur = arêtes portées par un seul triangle (les interfaces en ont deux)
    edge_basis = FacetBasis(m, ElementTriP1(), facets=m.boundary_facets())
    e_edge = edge_basis.with_element(ElementTriP0()).interpolate(fields["e"])
    A = A + _edge_lhs.assemble(edge_basis, e=e_edge, h=assembly.h)
    b = b + _edge_rhs.assemble(edge_basis, e=e_edge, h=assembly.h, T_inf=assembly.T_inf)

    return HeatProblem(basis=basis, edge_basis=edge_basis, fields=fields, A=A.tocsr(), b=b)


def heat_balance(assembly: Assembly, problem: HeatProblem, T: np.ndarray) -> dict[str, float]:
    """Bilan d'énergie [W] : puissance injectée, pertes par les faces et par la tranche.

    En régime stationnaire, injected = faces + edges. Pour la solution éléments
    finis, l'égalité est exacte (à l'erreur d'arrondi près), car v = 1 est une
    fonction test P1 : c'est un bon test de l'assemblage.
    """
    fields = problem.fields
    ones = np.ones(problem.basis.N)
    p0 = problem.basis.with_element(ElementTriP0())
    q = p0.interpolate(fields["q"])
    injected = LinearForm(lambda v, w: w.q * v).assemble(problem.basis, q=q) @ ones

    dT = problem.basis.interpolate(T - assembly.T_inf)
    faces = LinearForm(lambda v, w: 2.0 * assembly.h * w.dT * v).assemble(problem.basis, dT=dT) @ ones

    eb = problem.edge_basis
    e_edge = eb.with_element(ElementTriP0()).interpolate(fields["e"])
    dT_edge = eb.interpolate(T - assembly.T_inf)
    edges = LinearForm(lambda v, w: assembly.h * w.e * w.dT * v).assemble(eb, e=e_edge, dT=dT_edge) @ ones

    return {"injected": float(injected), "faces": float(faces), "edges": float(edges)}


# Affichage
def plot_temperature(assembly: Assembly, mesh: Mesh, T: np.ndarray, ax=None, relative: bool = True,
                     show_mesh: bool = False, vmax: float | None = None):
    """Trace le champ de température sur l'assemblage.

    relative  : si vrai, affiche T - T_inf (l'élévation de température) ; sinon T.
    show_mesh : superpose les arêtes du maillage.
    vmax      : borne haute de l'échelle de couleurs (pour comparer deux tracés à la même échelle).
    Les contours des plaques sont en blanc ; la plaque source est marquée "(S)" et
    les pièces dont le maximum dépasse T_crit sont encadrées en rouge.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if ax is None:
        _, ax = plt.subplots()
    values = T - assembly.T_inf if relative else T
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    tpc = ax.tripcolor(x, y, mesh.triangles, values, shading="gouraud", cmap="inferno",
                       vmin=0.0 if relative else None, vmax=vmax)
    if show_mesh:
        ax.triplot(x, y, mesh.triangles, color="white", linewidth=0.15, alpha=0.5)

    for i, p in enumerate(assembly.plates):
        T_max = T[np.unique(mesh.triangles[mesh.piece == i])].max()
        hot = T_max > assembly.T_crit
        ax.add_patch(Rectangle((p.x, p.y), p.width, p.height, fill=False,
                               edgecolor="red" if hot else "white", linewidth=2.0 if hot else 0.8))
        label = f"{i}" + (" (S)" if p.q > 0 else "") + f"\nmax {T_max:.1f}"
        ax.text(p.x + p.width / 2, p.y + p.height / 2, label, ha="center", va="center", fontsize=7,
                color="white", bbox=dict(facecolor="black", alpha=0.4, linewidth=0, pad=1))

    ax.figure.colorbar(tpc, ax=ax, label="T - T_inf [K]" if relative else "T")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    return ax


if __name__ == "__main__":
    # Exemple : un même assemblage maillé tout fin puis avec des niveaux mélangés,
    # tracés à la même échelle de couleurs pour comparer.
    import matplotlib.pyplot as plt
    from skfem import solve

    from generator import generate_assembly
    from mesh import COARSE, FINE, ONE_NODE, build_mesh

    assembly = generate_assembly((4, 6), seed=0)
    n = len(assembly.plates)
    levels = np.random.default_rng(1).choice([FINE, COARSE, ONE_NODE], size=n).tolist()

    results = []
    for name, lv in [("tout fin", None), ("niveaux mélangés", levels)]:
        mesh = build_mesh(assembly, lv)
        problem = assemble_heat(assembly, mesh)
        T = solve(problem.A, problem.b)
        bal = heat_balance(assembly, problem, T)
        print(f"{name:18s}: {len(mesh.nodes):5d} nœuds, max(T) - T_inf = {T.max() - assembly.T_inf:.4f} K, "
              f"bilan injecté / perdu = {bal['injected']:.4g} / {bal['faces'] + bal['edges']:.4g} W")
        results.append((name, mesh, T))

    vmax = max(T.max() for _, _, T in results) - assembly.T_inf
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (name, mesh, T) in zip(axes, results):
        plot_temperature(assembly, mesh, T, ax=ax, show_mesh=True, vmax=vmax)
        ax.set_title(f"{name} : {len(mesh.nodes)} nœuds")
    plt.tight_layout()
    plt.show()
