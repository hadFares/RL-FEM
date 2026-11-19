from dataclasses import dataclass


@dataclass
class Plate:
    """Une plaque rectangulaire de l'assemblage."""

    x: float        # coin bas-gauche, abscisse [m]
    y: float        # coin bas-gauche, ordonnée [m]
    width: float    # largeur, selon x [m]
    height: float   # hauteur, selon y [m]
    k: float        # conductivité thermique du matériau [W/(m·K)]
    e: float        # épaisseur de la plaque [m]
    q: float = 0.0  # puissance surfacique de la source [W/m²] (0 si pas de source)


@dataclass
class Assembly:
    """Un assemblage de plaques en contact, refroidi par convection.

    L'indice d'une plaque dans `plates` sert d'identifiant :
    plates[i] correspond au groupe physique "piece_i" du maillage.
    """

    plates: list[Plate]
    h: float        # coefficient de convection avec l'air [W/(m²·K)]
    T_inf: float    # température de l'air ambiant [K ou °C]
    T_crit: float   # seuil critique de température [même unité que T_inf]
    seed: int       # graine aléatoire ayant servi à générer l'assemblage
