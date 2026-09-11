"""
tests/test_reproduction.py — Paper 1 (snapshot inmutable)
=========================================================
Verifica que este snapshot sigue reproduciendo los resultados publicados.

Tres niveles:
  1. AISLAMIENTO: los modulos se cargan desde `src/` de este snapshot y NO del
     codigo vivo del repositorio.
  2. INTEGRIDAD: SHA256SUMS coincide con el contenido del snapshot.
  3. REPRODUCCION: `reproduce/run_all.py --quick` regenera los numeros de
     `expected_outputs/quick/numbers.json` dentro de tolerancia.

TOLERANCIAS. El modelo es determinista y no usa numeros aleatorios. En la misma
maquina y con el mismo NumPy la reproduccion es bit a bit. Entre maquinas, el
orden de las operaciones de punto flotante puede diferir (BLAS, SIMD), asi que
se admite 1e-9 relativo en las magnitudes continuas. Las magnitudes que deben
ser EXACTAMENTE cero (well-balanced) se comprueban como cero exacto: si dejaran
de serlo, seria un fallo real, no un artefacto de plataforma.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

SNAPSHOT = Path(__file__).resolve().parents[1]
SRC = SNAPSHOT / "src"
ESPERADO_QUICK = SNAPSHOT / "expected_outputs" / "quick" / "numbers.json"
ESPERADO_FULL = SNAPSHOT / "expected_outputs" / "numbers.json"

RTOL = 1e-9


# ---------------------------------------------------------------------------
# 1. Aislamiento
# ---------------------------------------------------------------------------
def test_los_modulos_se_cargan_del_snapshot_no_del_codigo_vivo():
    """
    El re-enraizado de imports debe ganar sobre cualquier otra entrada de
    sys.path. Si esto falla, el snapshot no esta congelado y no prueba nada.
    """
    import constants.titan_params as tp
    import core.native_fvm as nf
    import core.native_fvm2d as nf2
    for mod in (tp, nf, nf2):
        ruta = Path(mod.__file__).resolve()
        assert SRC in ruta.parents, (
            f"{mod.__name__} se cargo de {ruta}, no de {SRC}. "
            f"El snapshot NO esta aislado del codigo vivo.")


def test_el_snapshot_contiene_todos_los_paquetes():
    for paquete in ("constants", "core", "domain", "forcing", "physics",
                    "titan_io", "utils", "validation"):
        assert (SRC / paquete / "__init__.py").is_file(), paquete


def test_el_snapshot_tiene_las_piezas_del_protocolo_a():
    for pieza in ("requirements.txt", "README.md", "PROVENANCE.md", "LICENSE",
                  "CITATION.cff", "SHA256SUMS", "reproduce/run_all.py"):
        assert (SNAPSHOT / pieza).exists(), pieza


# ---------------------------------------------------------------------------
# 2. Integridad
# ---------------------------------------------------------------------------
def _parsear_sums(texto: str) -> list[tuple[str, str]]:
    """
    Parsea un SHA256SUMS tolerando los DOS formatos de coreutils y cualquier
    fin de linea.

        <digest>  <ruta>     modo texto   (dos espacios)
        <digest> *<ruta>     modo binario (un espacio y asterisco)

    POR QUE ES TOLERANTE. `sha256sum` escribe en modo texto o en modo binario
    segun como se invoque, y un manifiesto regenerado con `-b` usa el segundo.
    Partir por dos espacios revienta con `ValueError: not enough values to
    unpack` sobre un manifiesto binario, y entonces el test de integridad
    fallaria mientras `sha256sum -c` da todo OK. Partir por espacio en blanco
    generico y quitar el asterisco acepta ambos formatos sin relajar nada: el
    digest se sigue comparando byte a byte.
    """
    entradas = []
    for linea in texto.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        linea = linea.strip()
        if not linea:
            continue
        partes = linea.split(None, 1)
        if len(partes) != 2:
            raise ValueError(f"linea de SHA256SUMS no parseable: {linea!r}")
        digest, ruta = partes
        entradas.append((digest.strip().lower(), ruta.lstrip("*").strip()))
    return entradas


def _comprobar_checksums(base: Path) -> tuple[list[str], list[str], int]:
    """Devuelve (fallos, ausentes, comprobados) para el snapshot en `base`."""
    fallos, ausentes, comprobados = [], [], 0
    for h, rel in _parsear_sums((base / "SHA256SUMS").read_text(encoding="utf-8")):
        f = base / rel
        if not f.is_file():
            ausentes.append(rel)
        else:
            comprobados += 1
            if hashlib.sha256(f.read_bytes()).hexdigest() != h:
                fallos.append(rel)
    return fallos, ausentes, comprobados


def test_los_checksums_coinciden():
    """
    SHA256SUMS se genera al cerrar el snapshot. Cualquier discrepancia significa
    que alguien lo ha modificado despues de congelarlo, lo que esta prohibido.

    Los directorios de salida regenerables (outputs/, outputs_quick/) se excluyen
    porque los produce la propia reproduccion.
    """
    fallos, ausentes, comprobados = _comprobar_checksums(SNAPSHOT)
    assert comprobados > 0, (
        "no se comprobo NI UN fichero: el manifiesto no se pudo leer. Una "
        "verificacion sin lecturas no es una verificacion que pasa.")
    assert not ausentes, f"ficheros ausentes: {ausentes}"
    assert not fallos, f"ficheros modificados tras congelar: {fallos}"


def test_el_parser_de_checksums_acepta_los_dos_formatos():
    """El formato binario (`<digest> *ruta`) es el que rompia este test."""
    texto = ("aa" * 32 + "  ruta/en/modo/texto.py\n"
             + "bb" * 32 + " *ruta/en/modo/binario.py\n")
    assert _parsear_sums(texto) == [("aa" * 32, "ruta/en/modo/texto.py"),
                                    ("bb" * 32, "ruta/en/modo/binario.py")]
    # Y con finales CRLF, que es como se escribian antes los manifiestos.
    assert _parsear_sums(texto.replace("\n", "\r\n")) == _parsear_sums(texto)


def test_la_comprobacion_de_checksums_detecta_un_byte_alterado(tmp_path):
    """
    CONTROL NEGATIVO. Un verificador al que no se ha visto fallar no es un
    verificador. Se construye un snapshot de juguete, se altera un byte
    y se exige que la comprobacion lo cace, en los dos formatos de manifiesto.
    """
    # `marca` es el separador del manifiesto: dos espacios (modo texto) o
    # espacio + asterisco (modo binario). El nombre del directorio NO puede
    # derivarse de ella: un `*` es ilegal en una ruta de Windows.
    for i, marca in enumerate(("  ", " *")):
        base = tmp_path / f"snap{i}"
        (base / "src").mkdir(parents=True)
        f = base / "src" / "modulo.py"
        f.write_bytes(b"G = 1.352\n")
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        (base / "SHA256SUMS").write_text(f"{h}{marca}src/modulo.py\n",
                                         encoding="utf-8", newline="\n")

        fallos, ausentes, comprobados = _comprobar_checksums(base)
        assert (fallos, ausentes, comprobados) == ([], [], 1), marca

        f.write_bytes(b"G = 9.810\n")          # un valor fisico alterado
        fallos, ausentes, comprobados = _comprobar_checksums(base)
        assert fallos == ["src/modulo.py"], f"no detecto la alteracion ({marca!r})"
        assert comprobados == 1

        f.unlink()                              # y un fichero desaparecido
        fallos, ausentes, comprobados = _comprobar_checksums(base)
        assert ausentes == ["src/modulo.py"], f"no detecto la ausencia ({marca!r})"


# ---------------------------------------------------------------------------
# 3. Reproduccion
# ---------------------------------------------------------------------------
def _comparar(obtenido, esperado, ruta=""):
    """Compara recursivamente dos estructuras JSON. Devuelve lista de fallos."""
    fallos = []
    if isinstance(esperado, dict):
        for k, v in esperado.items():
            if k == "meta":
                continue
            if k not in obtenido:
                fallos.append(f"{ruta}/{k}: ausente")
            else:
                fallos += _comparar(obtenido[k], v, f"{ruta}/{k}")
    elif isinstance(esperado, list):
        if len(obtenido) != len(esperado):
            fallos.append(f"{ruta}: longitud {len(obtenido)} != {len(esperado)}")
        else:
            for i, v in enumerate(esperado):
                fallos += _comparar(obtenido[i], v, f"{ruta}[{i}]")
    elif isinstance(esperado, (int, float)) and not isinstance(esperado, bool):
        o = obtenido
        if esperado == 0.0:
            if o != 0.0:
                fallos.append(f"{ruta}: {o!r} deberia ser CERO EXACTO")
        elif abs(o - esperado) > RTOL * abs(esperado):
            fallos.append(f"{ruta}: {o!r} != {esperado!r} "
                          f"(rel {abs(o-esperado)/abs(esperado):.2e})")
    elif obtenido != esperado:
        fallos.append(f"{ruta}: {obtenido!r} != {esperado!r}")
    return fallos


@pytest.mark.skipif(not ESPERADO_QUICK.is_file(),
                    reason="falta expected_outputs/quick/numbers.json")
def test_la_reproduccion_rapida_da_los_numeros_esperados(tmp_path):
    r = subprocess.run(
        [sys.executable, str(SNAPSHOT / "reproduce" / "run_all.py"),
         "--quick", "--outdir", str(tmp_path)],
        capture_output=True, text=True, cwd=str(SNAPSHOT))
    assert r.returncode == 0, r.stdout[-4000:] + "\n" + r.stderr[-4000:]

    obtenido = json.loads((tmp_path / "numbers.json").read_text(encoding="utf-8"))
    esperado = json.loads(ESPERADO_QUICK.read_text(encoding="utf-8"))
    fallos = _comparar(obtenido, esperado)
    assert not fallos, "\n".join(fallos[:40])


@pytest.mark.skipif(not ESPERADO_FULL.is_file(),
                    reason="falta expected_outputs/numbers.json")
def test_los_numeros_publicados_cumplen_las_compuertas():
    """
    Comprobacion directa sobre los numeros de referencia del paper: cada
    compuerta de validacion debe seguir cumpliendose en el fichero publicado.
    Es independiente de volver a correr el modelo.
    """
    n = json.loads(ESPERADO_FULL.read_text(encoding="utf-8"))

    # Compuerta 1: cero EXACTO.
    g1 = n["gate1_well_balanced"]
    for k in ("zeta_max_1d", "u_max_1d", "zeta_max_2d", "u_max_2d", "v_max_2d",
              "mass_rel_error_1d", "mass_rel_error_2d"):
        assert g1[k] == 0.0, f"{k} = {g1[k]!r}, deberia ser cero exacto"

    # Compuerta 2: conservacion.
    g2 = n["gate2_conservation"]
    assert g2["mass_drift_1d"] < 1e-12
    assert g2["energy_growth_1d"] < 1e-6

    # Compuerta 3: dam-break, orden ~1 por el choque.
    assert n["gate3_dambreak"]["l1_order"] >= 0.8

    # Compuerta 4: orden formal 2 del esquema base.
    g4 = n["gate4_mms"]
    assert g4["order_1d_none"] >= 1.90
    assert g4["order_2d_none"] >= 1.90
    assert g4["order_2d_minmod"] >= 1.40
    assert 0.7 < g4["order_2d_order1"] < 1.4, "el control negativo fallo"

    # Barometro inverso.
    assert n["inverse_barometer"]["rel_error"] < 0.02

    # Compuerta 5: pico en F = 1 y convergencia de malla.
    g5 = n["gate5_sweep"]
    assert g5["peak_speed"] == pytest.approx(g5["resonance_speed"], rel=1e-9)
    assert g5["peak_amplification"] > 8.0
    picos = g5["peak_by_resolution"]
    assert all(b > a for a, b in zip(picos, picos[1:])), (
        "el pico deberia crecer al refinar")
    assert g5["peak_convergence_ratio"] > 1.5
    assert g5["peak_richardson"] > picos[-1]

    # Consistencia 2D <-> 1D a precision de maquina.
    assert n["consistency_2d_1d"]["rel_error_same_dt"] < 1e-12
    assert n["consistency_2d_1d"]["v_max_2d"] == 0.0


@pytest.mark.skipif(not ESPERADO_FULL.is_file(), reason="falta numbers.json")
def test_los_numeros_publicados_declaran_su_procedencia_pendiente():
    """
    Un fichero de resultados sin la lista TO_VERIFY no es utilizable para un
    manuscrito. Mientras OBSERVED_FRONT_SPEED siga sin verificar, la
    localizacion de la banda de profundidad resonante no es publicable.
    """
    meta = json.loads(ESPERADO_FULL.read_text(encoding="utf-8"))["meta"]
    assert meta["to_verify"], "falta la lista de constantes TO_VERIFY"
    assert any("OBSERVED_FRONT_SPEED" in s for s in meta["to_verify"])
    assert meta["observed_front_speed"] == [2.0, 10.0]
    assert meta["sweep_speed_range"] == [2.0, 20.0]


@pytest.mark.skipif(not ESPERADO_FULL.is_file(), reason="falta numbers.json")
def test_las_figuras_del_paper_existen():
    for fig in ("fig1_dambreak_profile.png", "fig2_mms_convergence.png",
                "fig3_amplification_curve.png",
                "fig4_peak_mesh_convergence.png"):
        f = SNAPSHOT / "expected_outputs" / fig
        assert f.is_file() and f.stat().st_size > 1000, fig
