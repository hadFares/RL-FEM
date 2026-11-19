"""Générateur de géométrie : des plaques rectangulaires accrochées les unes aux autres.

Principe ("croissance par accrochage") :
  1. on tire le nombre de plaques n dans la range demandée ;
  2. on pose une première plaque au hasard dans le carré [0, DOMAIN]² ;
  3. chaque nouvelle plaque est collée contre un côté d'une plaque déjà posée,
     avec un décalage aléatoire ; on la garde si elle est valide (dans le carré,
     sans chevauchement, sans contact trop court) ;
  4. si on n'arrive pas à placer une plaque, on recommence tout l'assemblage.

Un rectangle est un tuple d'entiers (x, y, w, h) : coin bas-gauche, largeur, hauteur.
Coordonnées entières pour que les bords en contact coïncident exactement
(sinon `fragment` de Gmsh ne recolle pas les plaques).
"""

import numpy as np

from assembly import Assembly, Plate

Rect = tuple[int, int, int, int]  # (x, y, w, h)

# Paramètres de la géométrie (espace normalisé)
DOMAIN = 100          # le système vit dans le carré [0, 100] x [0, 100]
MIN_SIDE = 10         # plus petit côté autorisé pour une plaque
MAX_ASPECT = 4        # rapport longueur / largeur maximal d'une plaque
AREA_FACTOR = 1.5     # aire max d'une plaque = AREA_FACTOR * DOMAIN² / n
MIN_CONTACT = 5       # longueur minimale d'un contact entre deux plaques

MAX_TRIES_PER_PLATE = 100   # essais pour placer une plaque avant d'abandonner
MAX_RESTARTS = 1000         # nombre de redémarrages avant de lever une erreur

# Plages des propriétés physiques (PROVISOIRES : à calibrer quand le solveur tournera)
K_RANGE = (0.2, 400.0)        # conductivité, tirée en log-uniforme (plastique -> cuivre)
E_RANGE = (1.0, 5.0)          # épaisseur
Q_RANGE = (1.0, 10.0)         # puissance surfacique de la source
H_RANGE = (5.0, 50.0)         # coefficient de convection
T_INF_RANGE = (15.0, 35.0)    # température de l'air
DT_CRIT_RANGE = (20.0, 80.0)  # T_crit = T_inf + un écart tiré dans cette plage


# Tests géométriques entre deux rectangles
def _overlap(a: Rect, b: Rect) -> bool:
    """Vrai si les intérieurs de a et b se chevauchent (se toucher n'est pas chevaucher)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def contact_length(a: Rect, b: Rect) -> int | None:
    """Longueur du bord commun entre a et b (supposés sans chevauchement).

    Renvoie None si a et b ne se touchent pas, 0 s'ils se touchent par un coin,
    et la longueur du segment commun sinon.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if ax + aw == bx or bx + bw == ax:          # un bord vertical commun
        length = min(ay + ah, by + bh) - max(ay, by)
    elif ay + ah == by or by + bh == ay:        # un bord horizontal commun
        length = min(ax + aw, bx + bw) - max(ax, bx)
    else:
        return None
    return length if length >= 0 else None


def _is_valid(new: Rect, plates: list[Rect]) -> bool:
    """Vrai si `new` peut être ajouté à l'assemblage `plates`."""
    x, y, w, h = new
    if x < 0 or y < 0 or x + w > DOMAIN or y + h > DOMAIN:
        return False
    for p in plates:
        if _overlap(new, p):
            return False
        c = contact_length(new, p)
        if c is not None and c < MIN_CONTACT:   # contact par un coin, ou trop court
            return False
    return True


# Tirages aléatoires
def _draw_size(rng: np.random.Generator, max_area: float) -> tuple[int, int]:
    """Tire une largeur et une hauteur entières respectant les bornes."""
    while True:
        w = int(rng.integers(MIN_SIDE, DOMAIN + 1))
        h = int(rng.integers(MIN_SIDE, DOMAIN + 1))
        if w * h <= max_area and max(w, h) <= MAX_ASPECT * min(w, h):
            return w, h


def _draw_attached(rng: np.random.Generator, anchor: Rect, w: int, h: int) -> Rect:
    """Place un rectangle w x h contre un côté de `anchor`, choisi au hasard.

    Le décalage le long du côté est tiré pour que le bord commun
    mesure au moins MIN_CONTACT.
    """
    ax, ay, aw, ah = anchor
    side = rng.integers(4)
    if side in (0, 1):  # à droite ou à gauche : on glisse selon y
        y = int(rng.integers(ay + MIN_CONTACT - h, ay + ah - MIN_CONTACT + 1))
        x = ax + aw if side == 0 else ax - w
    else:               # en haut ou en bas : on glisse selon x
        x = int(rng.integers(ax + MIN_CONTACT - w, ax + aw - MIN_CONTACT + 1))
        y = ay + ah if side == 2 else ay - h
    return (x, y, w, h)


def _try_build(rng: np.random.Generator, n: int) -> list[Rect] | None:
    """Une tentative de construction de n plaques. None si on reste bloqué."""
    max_area = AREA_FACTOR * DOMAIN**2 / n

    # Première plaque : n'importe où dans le carré
    w, h = _draw_size(rng, max_area)
    x = int(rng.integers(0, DOMAIN - w + 1))
    y = int(rng.integers(0, DOMAIN - h + 1))
    plates = [(x, y, w, h)]

    # Plaques suivantes : accrochées à une plaque existante
    while len(plates) < n:
        for _ in range(MAX_TRIES_PER_PLATE):
            anchor = plates[rng.integers(len(plates))]
            w, h = _draw_size(rng, max_area)
            new = _draw_attached(rng, anchor, w, h)
            if _is_valid(new, plates):
                plates.append(new)
                break
        else:  # aucun essai n'a marché
            return None
    return plates


# Fonction principale
def generate_geometry(n_range: tuple[int, int], rng: np.random.Generator) -> list[Rect]:
    """Génère un assemblage de plaques rectangulaires connexe.

    n_range : (n_min, n_max), bornes incluses, pour le nombre de plaques.
    rng     : générateur aléatoire numpy ; tous les tirages passent par lui,
              donc une même graine redonne le même assemblage.
    """
    n_min, n_max = n_range
    n = int(rng.integers(n_min, n_max + 1))
    for _ in range(MAX_RESTARTS):
        plates = _try_build(rng, n)
        if plates is not None:
            return plates
    raise RuntimeError(f"Impossible de placer {n} plaques après {MAX_RESTARTS} essais.")


def generate_assembly(n_range: tuple[int, int], seed: int) -> Assembly:
    """Génère un assemblage complet : géométrie + propriétés physiques aléatoires.

    Une seule plaque, tirée au hasard, porte la source (q > 0).
    La même graine redonne exactement le même assemblage.
    """
    rng = np.random.default_rng(seed)
    rects = generate_geometry(n_range, rng)
    source = int(rng.integers(len(rects)))

    plates = []
    for i, (x, y, w, h) in enumerate(rects):
        k = float(np.exp(rng.uniform(np.log(K_RANGE[0]), np.log(K_RANGE[1]))))
        e = float(rng.uniform(*E_RANGE))
        q = float(rng.uniform(*Q_RANGE)) if i == source else 0.0
        plates.append(Plate(x=x, y=y, width=w, height=h, k=k, e=e, q=q))

    T_inf = float(rng.uniform(*T_INF_RANGE))
    return Assembly(
        plates=plates,
        h=float(rng.uniform(*H_RANGE)),
        T_inf=T_inf,
        T_crit=T_inf + float(rng.uniform(*DT_CRIT_RANGE)),
        seed=seed,
    )


# Affichage
def plot_geometry(plates: list[Rect], ax=None):
    """Dessine les plaques (numérotées) dans le carré du domaine."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if ax is None:
        _, ax = plt.subplots()
    for i, (x, y, w, h) in enumerate(plates):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=f"C{i % 10}", edgecolor="black", alpha=0.7))
        ax.text(x + w / 2, y + h / 2, str(i), ha="center", va="center")
    ax.set_xlim(0, DOMAIN)
    ax.set_ylim(0, DOMAIN)
    ax.set_aspect("equal")
    return ax


if __name__ == "__main__":
    # Vérification visuelle : 20 assemblages tirés avec les graines 0 à 19
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    for seed, ax in enumerate(axes.flat):
        plates = generate_geometry((3, 5), np.random.default_rng(seed))
        plot_geometry(plates, ax)
        ax.set_title(f"graine {seed} : {len(plates)} plaques")
    plt.tight_layout()
    plt.show()
