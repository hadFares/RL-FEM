"""Maillage d'un assemblage avec Gmsh.

Chaque plaque devient une surface, avec son groupe physique "piece_i"
et sa propre taille de maille selon son niveau (fin, grossier, un nœud).
`fragment` recolle les plaques : les interfaces sont conformes (nœuds partagés).
L'air n'est pas maillé : la convection est un terme de l'équation sur les plaques.
"""

from dataclasses import dataclass
from pathlib import Path

import gmsh
import numpy as np

from assembly import Assembly
from generator import contact_length

# Niveaux de maillage d'une plaque (c'est l'action de la politique)
FINE, COARSE, ONE_NODE = 0, 1, 2
LEVEL_NAMES = {FINE: "fin", COARSE: "grossier", ONE_NODE: "un nœud"}

# Taille de maille de chaque niveau (espace normalisé 0-100, plus petit côté = 10)
LEVEL_SIZES = {
    FINE: 1.0,        # ~5 mailles sur le plus petit contact (5)
    COARSE: 5.0,      # ~2 mailles sur le plus petit côté (10)
    ONE_NODE: 1000.0, # plus grand que tout le domaine : maillage le plus pauvre possible
}


@dataclass
class Mesh:
    """Maillage en mémoire, prêt pour le solveur."""

    nodes: np.ndarray      # (N, 2) coordonnées des nœuds
    triangles: np.ndarray  # (M, 3) indices (dans `nodes`) des 3 sommets de chaque triangle
    piece: np.ndarray      # (M,) indice de la plaque à laquelle appartient chaque triangle


def build_mesh(assembly: Assembly, levels: list[int] | None = None, path: str | Path | None = None) -> Mesh:
    """Maille l'assemblage et renvoie le maillage en mémoire.

    levels : niveau de chaque plaque (FINE, COARSE ou ONE_NODE). Par défaut : tout fin.
    path   : si donné, écrit aussi le maillage dans ce fichier .msh.
    """
    n = len(assembly.plates)
    if levels is None:
        levels = [FINE] * n
    if len(levels) != n:
        raise ValueError(f"{len(levels)} niveaux pour {n} plaques")

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 2)  # n'afficher que avertissements et erreurs
        gmsh.model.add("assembly")
        surfaces = _build_geometry(assembly)
        _set_sizes(surfaces, levels)
        gmsh.model.mesh.generate(2)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            gmsh.write(str(path))
        return _extract(surfaces)
    finally:
        gmsh.finalize()


def _build_geometry(assembly: Assembly) -> list[int]:
    """Crée les rectangles, les recolle et les étiquette. Renvoie le tag de surface de chaque plaque."""
    tags = [gmsh.model.occ.addRectangle(p.x, p.y, 0, p.width, p.height) for p in assembly.plates]
    dimtags = [(2, t) for t in tags]

    # Recoller toutes les plaques. 'correspondance[i]' = ce qu'est devenue la plaque i.
    if len(dimtags) > 1:
        _, correspondance = gmsh.model.occ.fragment(dimtags[:1], dimtags[1:])
    else:
        correspondance = [dimtags]
    gmsh.model.occ.synchronize()

    surfaces = []
    for i, morceaux in enumerate(correspondance):
        assert len(morceaux) == 1, f"la plaque {i} a été coupée en {len(morceaux)} morceaux"
        s = morceaux[0][1]
        gmsh.model.addPhysicalGroup(2, [s], name=f"piece_{i}")
        surfaces.append(s)
    return surfaces


def _set_sizes(surfaces: list[int], levels: list[int]) -> None:
    """Une taille de maille par plaque : un champ Constant par plaque, combinés par un Min."""
    fields = []
    for s, level in zip(surfaces, levels):
        f = gmsh.model.mesh.field.add("Constant")
        gmsh.model.mesh.field.setNumbers(f, "SurfacesList", [s])
        gmsh.model.mesh.field.setNumber(f, "VIn", LEVEL_SIZES[level])
        gmsh.model.mesh.field.setNumber(f, "VOut", 1e22)
        gmsh.model.mesh.field.setNumber(f, "IncludeBoundary", 1)
        fields.append(f)

    # Sur une interface, c'est la plus petite des deux tailles qui l'emporte
    f_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

    # Seul notre champ décide de la taille
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)


def _extract(surfaces: list[int]) -> Mesh:
    """Récupère nœuds et triangles depuis Gmsh, sous forme de tableaux numpy."""
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    nodes = coords.reshape(-1, 3)[:, :2]

    # Les tags Gmsh des nœuds ne sont pas forcément 0, 1, 2... : on les convertit en indices
    index = np.full(node_tags.max() + 1, -1, dtype=int)
    index[node_tags] = np.arange(len(node_tags))

    triangles, piece = [], []
    for i, s in enumerate(surfaces):
        _, tri_nodes = gmsh.model.mesh.getElementsByType(2, tag=s)  # type 2 = triangle à 3 nœuds
        tri = index[tri_nodes].reshape(-1, 3)
        triangles.append(tri)
        piece.append(np.full(len(tri), i))
    return Mesh(nodes=nodes, triangles=np.vstack(triangles), piece=np.concatenate(piece))


def check_contacts(assembly: Assembly, mesh: Mesh) -> list[tuple[int, int, int, int]]:
    """Vérifie que les plaques en contact partagent des nœuds, et les autres non.

    Lève une AssertionError sinon. Renvoie (i, j, longueur du contact, nœuds partagés)
    pour chaque paire de plaques en contact.
    """
    rects = [(p.x, p.y, p.width, p.height) for p in assembly.plates]
    nodes_of = [set(np.unique(mesh.triangles[mesh.piece == i])) for i in range(len(rects))]
    contacts = []
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            c = contact_length(rects[i], rects[j])
            shared = len(nodes_of[i] & nodes_of[j])
            if c is None:
                assert shared == 0, f"plaques {i} et {j} séparées mais {shared} nœuds partagés"
            else:
                assert shared >= 2, f"contact {i}-{j} (longueur {c}) sans interface conforme"
                contacts.append((i, j, c, shared))
    return contacts


def plot_mesh(mesh: Mesh, ax=None, labels: list[str] | None = None):
    """Dessine le maillage, une couleur par plaque, avec un texte au centre de chaque plaque."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    ax.tripcolor(x, y, mesh.triangles, facecolors=mesh.piece % 20, cmap="tab20", vmin=0, vmax=19, alpha=0.6)
    ax.triplot(x, y, mesh.triangles, color="black", linewidth=0.2)
    for i in np.unique(mesh.piece):
        center = mesh.nodes[np.unique(mesh.triangles[mesh.piece == i])].mean(axis=0)
        text = labels[i] if labels is not None else str(i)
        ax.text(*center, text, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    return ax


if __name__ == "__main__":
    # Exemple : un assemblage moyen, maillé tout fin puis avec des niveaux mélangés
    import matplotlib.pyplot as plt

    from generator import generate_assembly

    assembly = generate_assembly((8, 10), seed=0)
    n = len(assembly.plates)

    fine = build_mesh(assembly, path="meshes/exemple_fin.msh")
    levels = np.random.default_rng(1).integers(3, size=n).tolist()
    mixed = build_mesh(assembly, levels)

    for name, m in [("tout fin", fine), ("niveaux mélangés", mixed)]:
        contacts = check_contacts(assembly, m)
        print(f"{name} : {len(m.nodes)} nœuds, {len(m.triangles)} triangles, "
              f"{len(contacts)} contacts conformes")
    print("Contacts (plaque i, plaque j, longueur, nœuds partagés) en tout fin :")
    for c in check_contacts(assembly, fine):
        print("  ", c)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    plot_mesh(fine, axes[0])
    axes[0].set_title(f"Tout fin : {len(fine.nodes)} nœuds")
    short = {FINE: "F", COARSE: "G", ONE_NODE: "1"}
    plot_mesh(mixed, axes[1], labels=[f"{i} ({short[lv]})" for i, lv in enumerate(levels)])
    axes[1].set_title(f"Niveaux mélangés (F = fin, G = grossier, 1 = un nœud) : {len(mixed.nodes)} nœuds")
    plt.tight_layout()
    plt.show()
