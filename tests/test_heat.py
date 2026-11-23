"""Vérifications de l'équation de la chaleur (src/heat.py).

Lancer avec : uv run pytest tests/test_heat.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from skfem import solve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assembly import Assembly, Plate  # noqa: E402
from generator import generate_assembly  # noqa: E402
from heat import LENGTH_UNIT, assemble_heat, heat_balance  # noqa: E402
from mesh import COARSE, build_mesh  # noqa: E402


def _solve(assembly, levels=None):
    mesh = build_mesh(assembly, levels)
    problem = assemble_heat(assembly, mesh)
    return problem, solve(problem.A, problem.b)


def test_uniform_plate_matches_lumped_balance():
    """Plaque très conductrice : T quasi uniforme, donnée par le bilan global
    q·S = (T - T_inf)·(2h·S + h·e·P)   (S = aire, P = périmètre)."""
    w, h_, e = 40, 20, 3
    plate = Plate(x=10, y=10, width=w, height=h_, k=1e6, e=e, q=5000.0)
    asm = Assembly(plates=[plate], h=20.0, T_inf=25.0, T_crit=100.0, seed=0)
    _, T = _solve(asm, [COARSE])

    S = w * h_ * LENGTH_UNIT**2
    P = 2 * (w + h_) * LENGTH_UNIT
    T_expected = asm.T_inf + plate.q * S / (2 * asm.h * S + asm.h * e * LENGTH_UNIT * P)
    assert np.ptp(T) < 1e-3
    assert T.mean() == pytest.approx(T_expected, rel=1e-6)


def test_no_source_gives_ambient_temperature():
    """Sans source, la solution est T = T_inf partout."""
    asm = generate_assembly((3, 5), seed=1)
    for p in asm.plates:
        p.q = 0.0
    _, T = _solve(asm, [COARSE] * len(asm.plates))
    np.testing.assert_allclose(T, asm.T_inf, atol=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_energy_balance(seed):
    """Puissance injectée = pertes faces + pertes tranche (exact pour la solution EF)."""
    asm = generate_assembly((3, 6), seed=seed)
    problem, T = _solve(asm, [COARSE] * len(asm.plates))
    bal = heat_balance(asm, problem, T)
    assert bal["injected"] > 0
    assert bal["faces"] + bal["edges"] == pytest.approx(bal["injected"], rel=1e-9)
    assert T.min() >= asm.T_inf - 1e-9  # principe du maximum : rien n'est plus froid que l'air
