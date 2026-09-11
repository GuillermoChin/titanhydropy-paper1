"""La fuente unica de constantes es coherente e importable."""

from __future__ import annotations

import math

import pytest

from constants.titan_params import (DEFAULT_FLUID, OBSERVED_FRONT_SPEED,
                                    REFERENCE_DEPTHS, SWEEP_SPEED_RANGE, TITAN,
                                    TO_VERIFY, audit_provenance,
                                    inverse_barometer, resonant_depth,
                                    resonant_depth_band, sanity_check,
                                    shallow_water_speed)
from validation.titan_physical import proudman_overlap


def test_sanity_check_pasa():
    sanity_check()


def test_omega_y_coriolis_coherentes():
    assert TITAN.omega == pytest.approx(2 * math.pi / TITAN.rotation_period)
    assert TITAN.coriolis(0.0) == pytest.approx(0.0, abs=1e-18)
    assert TITAN.coriolis(90.0) == pytest.approx(2 * TITAN.omega)
    # Antisimetria hemisferica.
    assert TITAN.coriolis(-78.0) == pytest.approx(-TITAN.coriolis(78.0))


def test_shallow_water_speed_rechaza_profundidad_no_positiva():
    with pytest.raises(ValueError):
        shallow_water_speed(0.0)


def test_signo_del_barometro_inverso():
    """Caida de presion (dP<0) => ascenso de superficie (zeta>0)."""
    assert inverse_barometer(-100.0) > 0.0
    assert inverse_barometer(+100.0) < 0.0
    assert inverse_barometer(-100.0) == pytest.approx(
        100.0 / (DEFAULT_FLUID.rho * TITAN.g))


def test_auditoria_de_procedencia_no_vacia():
    """Hay valores TO_VERIFY en uso: la auditoria debe delatarlos."""
    pending = audit_provenance()
    assert pending, "audit_provenance() no puede estar vacia todavia"
    assert any("OBSERVED_FRONT_SPEED" in p for p in pending)
    assert OBSERVED_FRONT_SPEED[2] == TO_VERIFY


def test_el_barrido_no_es_un_dato_y_no_se_audita():
    """
    SWEEP_SPEED_RANGE es una decision de diseno del experimento: no lleva
    estatus de verificacion y NO debe aparecer en audit_provenance().
    """
    assert len(SWEEP_SPEED_RANGE) == 2, "el barrido no lleva estatus ni fuente"
    assert all(isinstance(x, float) for x in SWEEP_SPEED_RANGE)
    assert not any("SWEEP_SPEED_RANGE" in p for p in audit_provenance())
    # El dato SI lleva estatus y fuente.
    assert len(OBSERVED_FRONT_SPEED) == 4
    assert "10.1038/ngeo2406" in OBSERVED_FRONT_SPEED[3]


def test_banda_resonante_es_inversa_de_la_celeridad():
    """h_res(U) = U^2/g debe invertir exactamente c(h) = sqrt(g h)."""
    for u in (2.0, 7.5, 14.7, 20.0):
        assert shallow_water_speed(resonant_depth(u)) == pytest.approx(u)
    h_min, h_max = resonant_depth_band()
    assert h_min == pytest.approx(resonant_depth(OBSERVED_FRONT_SPEED[0]))
    assert h_max == pytest.approx(resonant_depth(OBSERVED_FRONT_SPEED[1]))


def test_linchpin_de_proudman_para_ligeia():
    """
    La banda resonante debe intersecar el rango de profundidades de Ligeia.

    NO se exige que c(h_max) caiga dentro del rango de frentes observados: con
    OBSERVED_FRONT_SPEED = [2,10] m/s eso es FALSO (c(Ligeia_max)=14.7 m/s) y
    sin embargo la resonancia si es alcanzable en los flancos intermedios de la
    cuenca.
    """
    ov = proudman_overlap()
    ligeia = ov["basins"]["ligeia_max"]
    assert ligeia["resonance_reachable"], (
        f"banda resonante {ov['resonant_depth_band_m']} m no interseca "
        f"[0, {ligeia['depth_m']}] m")
    # Y se documenta explicitamente que NO resuena a profundidad maxima.
    assert not ligeia["resonance_at_max_depth"], (
        "c(Ligeia_max) ha vuelto a caer dentro de OBSERVED_FRONT_SPEED: "
        "la resonancia dejaria de ser un fenomeno exclusivo de los flancos")


def test_todas_las_cuencas_de_referencia_alcanzan_la_resonancia():
    ov = proudman_overlap()
    for nombre, b in ov["basins"].items():
        assert b["resonance_reachable"], nombre


def test_profundidades_de_referencia_positivas_y_etiquetadas():
    for name, (depth, status, src) in REFERENCE_DEPTHS.items():
        assert depth > 0.0, name
        assert src, f"{name} sin fuente declarada"
        assert status in ("verified", "to_verify"), name
