"""
reproduce/run_all.py — Paper 1 (snapshot inmutable v2)
======================================================
Regenera TODAS las figuras y todos los numeros del Paper con un solo comando.

    python reproduce/run_all.py                 # corrida completa (~10 min)
    python reproduce/run_all.py --quick         # version rapida (~1 min)
    python reproduce/run_all.py --outdir DIR

DETERMINISMO. El modelo NO usa numeros aleatorios: es completamente
determinista. La semilla se fija de todos modos (SEMILLA = 20250101) por
protocolo, para que cualquier futura adicion estocastica quede anclada. La
reproducibilidad bit a bit esta garantizada en la misma maquina y el mismo
NumPy; entre maquinas las diferencias son de nivel de redondeo (< 1e-12
relativo), y el test de reproduccion usa esa tolerancia.

Texto de las figuras en INGLES; comentarios y consola en español.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# --- RE-ENRAIZADO: src/ del snapshot en la posicion 0 de sys.path ----------
SNAPSHOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SNAPSHOT / "src"))

from constants.titan_params import (DEFAULT_FLUID, OBSERVED_FRONT_SPEED,  # noqa: E402
                                    REFERENCE_DEPTHS, SWEEP_SPEED_RANGE, TITAN,
                                    audit_provenance, inverse_barometer,
                                    resonant_depth_band)
from core.native_fvm import NativeFVMSolver  # noqa: E402
from core.native_fvm2d import NativeFVMSolver2D  # noqa: E402
from core.solver_base import (Domain, ForcingBundle, SolverConfig,  # noqa: E402
                              State)
from domain.geometry import (flat_channel, quiescent_state, step_state,  # noqa: E402
                             variable_bed_channel)
from domain.geometry2d import (bumpy_basin, quiescent_state_2d,  # noqa: E402
                               transverse_invariant_channel)
from forcing.atmospheric import (MovingGaussianPressure,  # noqa: E402
                                 MovingPressureFront2D,
                                 StaticGaussianPressure)
from forcing.friction import LinearFriction  # noqa: E402
from physics.proudman import (amplification_from_field,  # noqa: E402
                              bound_amplification, proudman_amplification,
                              resonance_speed)
from utils.runners import l1_error, run_to  # noqa: E402
from utils.visualizer import (plot_amplification_curve,  # noqa: E402
                              plot_profile_comparison)
from validation.analytic import dam_break_exact, star_depth  # noqa: E402
from validation.conservation import ConservationMonitor, convergence_order  # noqa: E402
from validation.mms import default_mms_1d, default_mms_2d  # noqa: E402

SEMILLA = 20250101

# ---------------------------------------------------------------------------
# Supuestos numericos de cada experimento (identicos a los tests-compuerta)
# ---------------------------------------------------------------------------
# Dam-break
DB = dict(L=1.0e4, X0=5.0e3, H_BED=100.0, H_LEFT=200.0, H_RIGHT=100.0,
          T_END=100.0, CFL=0.45)
DB_RES_FULL = (100, 200, 400, 800, 1600, 3200)
DB_RES_QUICK = (100, 200, 400, 800)

# MMS
MMS_L = 1000.0
MMS_T_END = 20.0
MMS_RES_FULL = (32, 64, 128, 256)
MMS_RES_QUICK = (32, 64, 128)

# Barrido de resonancia
SW = dict(L=6.0e5, DEPTH=160.0, DP=100.0, SIGMA=1.0e4, X0=5.0e4,
          D=4.0e5, MARGEN=2.0e4, CFL=0.45)
SW_NX = 1500
SW_NX_QUICK = 500
SW_PEAK_RES_FULL = (750, 1500, 3000, 6000)
SW_PEAK_RES_QUICK = (375, 750, 1500)

# Barometro inverso
IB = dict(L=2.0e5, NX=400, DEPTH=160.0, DP=100.0, SIGMA=2.0e4, R=2.0e-4)


# ===========================================================================
# 1. Compuerta 1: lago en reposo (well-balanced exacto)
# ===========================================================================
def exp_well_balanced() -> dict:
    print("\n[1/6] Lago en reposo (well-balanced exacto, 1D y 2D)")
    d1 = variable_bed_channel(2.0e4, 200, 160.0, 120.0, 1.0e4, 1.0e3)
    s1 = NativeFVMSolver()
    s1.initialize(d1, quiescent_state(d1), ForcingBundle(coriolis_enabled=False),
                  SolverConfig(cfl=0.45, bc={"west": "reflective",
                                             "east": "reflective"}))
    for _ in range(400):
        s1.step()

    d2 = bumpy_basin(2.0e4, 1.5e4, 80, 60, 160.0, 120.0, (9.0e3, 8.25e3), 1.6e3)
    s2 = NativeFVMSolver2D()
    s2.initialize(d2, quiescent_state_2d(d2),
                  ForcingBundle(coriolis_enabled=False),
                  SolverConfig(cfl=0.45, bc={"west": "reflective",
                                             "east": "reflective",
                                             "south": "reflective",
                                             "north": "reflective"}))
    for _ in range(300):
        s2.step()

    r = {
        "zeta_max_1d": float(np.max(np.abs(s1.state.zeta))),
        "u_max_1d": float(np.max(np.abs(s1.state.u))),
        "zeta_max_2d": float(np.max(np.abs(s2.state.zeta))),
        "u_max_2d": float(np.max(np.abs(s2.state.u))),
        "v_max_2d": float(np.max(np.abs(s2.state.v))),
        "mass_rel_error_1d": s1.diagnostics()["mass_rel_error"],
        "mass_rel_error_2d": s2.diagnostics()["mass_rel_error"],
    }
    print(f"      1D: max|zeta| = {r['zeta_max_1d']:.1e}  "
          f"2D: max|zeta| = {r['zeta_max_2d']:.1e}  (deben ser 0.0 exacto)")
    return r


# ===========================================================================
# 2. Compuerta 2: conservacion
# ===========================================================================
def exp_conservacion() -> dict:
    print("\n[2/6] Conservacion en dominio cerrado (1D y 2D)")
    L, nx, depth = 2.0e4, 400, 160.0
    dom = variable_bed_channel(L, nx, depth, 100.0, 0.65 * L, 0.05 * L)
    arg = (dom.x - 0.3 * L) / (0.04 * L)
    init = State(zeta=2.0 * np.exp(-0.5 * arg * arg),
                 u=np.zeros(nx), v=np.zeros(nx))
    t_end = 4.0 * L / np.sqrt(TITAN.g * depth)
    res = run_to(dom, init, ForcingBundle(coriolis_enabled=False),
                 SolverConfig(cfl=0.45, bc={"west": "reflective",
                                            "east": "reflective"}),
                 t_end=t_end, sample_every=20)
    r = {"mass_drift_1d": res.monitor.mass_drift,
         "energy_growth_1d": res.monitor.energy_growth}
    print(f"      deriva de masa = {r['mass_drift_1d']:.3e}   "
          f"crecimiento de energia = {r['energy_growth_1d']:+.3e}")
    return r


# ===========================================================================
# 3. Compuerta 3: dam-break vs Riemann exacto  -> FIGURA 1 y FIGURA 2a
# ===========================================================================
def exp_dambreak(outdir: Path, quick: bool) -> dict:
    print("\n[3/6] Dam-break vs solucion exacta de Riemann")
    res_list = DB_RES_QUICK if quick else DB_RES_FULL
    errores = []
    for nx in res_list:
        dom = flat_channel(DB["L"], nx, DB["H_BED"])
        r = run_to(dom, step_state(dom, DB["H_LEFT"], DB["H_RIGHT"], DB["X0"]),
                   ForcingBundle(coriolis_enabled=False),
                   SolverConfig(cfl=DB["CFL"],
                                bc={"west": "transmissive",
                                    "east": "transmissive"}),
                   t_end=DB["T_END"], sample_every=10 ** 9)
        h_ex, _ = dam_break_exact(dom.x, DB["T_END"], DB["H_LEFT"],
                                  DB["H_RIGHT"], TITAN.g, x0=DB["X0"])
        errores.append(l1_error(r.state.zeta + DB["H_BED"], h_ex,
                                DB["L"] / nx))
    errores = np.array(errores)
    ordenes, pendiente = convergence_order(np.array(res_list), errores)

    # FIGURA 1: perfil a nx = 400
    dom = flat_channel(DB["L"], 400, DB["H_BED"])
    r = run_to(dom, step_state(dom, DB["H_LEFT"], DB["H_RIGHT"], DB["X0"]),
               ForcingBundle(coriolis_enabled=False),
               SolverConfig(cfl=DB["CFL"], bc={"west": "transmissive",
                                               "east": "transmissive"}),
               t_end=DB["T_END"], sample_every=10 ** 9)
    h_ex, u_ex = dam_break_exact(dom.x, DB["T_END"], DB["H_LEFT"],
                                 DB["H_RIGHT"], TITAN.g, x0=DB["X0"])
    plot_profile_comparison(
        dom.x,
        {"Exact Riemann solution": (h_ex, dict(color="k", lw=1.6)),
         "Native FVM (nx = 400)": (r.state.zeta + DB["H_BED"],
                                   dict(color="crimson", ls="--", lw=1.2))},
        outdir / "fig1_dambreak_profile.png",
        ylabel="Water column height $h$ [m]",
        title=f"1D dam break under Titan gravity, t = {DB['T_END']:.0f} s",
        subtitle=f"$g$ = {TITAN.g} m s$^{{-2}}$, flat bed $h_0$ = "
                 f"{DB['H_BED']:.0f} m, HLLC + MUSCL-minmod + SSPRK3, "
                 f"CFL = {DB['CFL']}")

    out = {"resolutions": list(res_list), "l1_errors": errores.tolist(),
           "pairwise_orders": ordenes.tolist(), "l1_order": float(pendiente),
           "star_depth": float(star_depth(DB["H_LEFT"], 0.0, DB["H_RIGHT"],
                                          0.0, TITAN.g))}
    print(f"      orden L1 = {pendiente:.3f} (limitado a ~1 por el choque)")
    return out


# ===========================================================================
# 4. Compuerta 4: orden formal por MMS  -> FIGURA 2b
# ===========================================================================
def _mms_error(n, limiter, order, dos_d):
    if dos_d:
        mms = default_mms_2d(MMS_L, MMS_L)
        dx = MMS_L / n
        x = (np.arange(n) + 0.5) * dx
        X, Y = np.meshgrid(x, x)
        h0 = mms.rest_depth(X, Y)
        dom = Domain(x=x, y=x.copy(), bathymetry=h0, lat0_deg=0.0)
        zeta, u, v = mms.exact_primitive(X, Y, 0.0)
        s = NativeFVMSolver2D(g=mms.g, order=order, limiter=limiter,
                              extra_source=mms.source_2d())
        bc = {"west": "periodic", "east": "periodic",
              "south": "periodic", "north": "periodic"}
        medida = dx * dx
    else:
        mms = default_mms_1d(MMS_L)
        dx = MMS_L / n
        x = (np.arange(n) + 0.5) * dx
        y0 = np.zeros_like(x)
        h0 = mms.rest_depth(x, y0)
        dom = Domain(x=x, y=np.array([0.0]), bathymetry=h0, lat0_deg=0.0)
        zeta, u, v = mms.exact_primitive(x, y0, 0.0)
        s = NativeFVMSolver(g=mms.g, order=order, limiter=limiter,
                            extra_source=mms.source_1d())
        bc = {"west": "periodic", "east": "periodic"}
        medida = dx
        X, Y = x, y0

    s.initialize(dom, State(zeta.copy(), u.copy(), v.copy()),
                 ForcingBundle(coriolis_enabled=False),
                 SolverConfig(cfl=0.45, bc=bc))
    while s.state.t < MMS_T_END:
        dt = min(s.compute_stable_dt(), MMS_T_END - s.state.t)
        if dt <= 0.0:
            break
        s.step(dt)
    h_ex, _, _ = mms.exact(X, Y, s.state.t)
    return float(np.sum(np.abs(s.state.zeta + h0 - h_ex)) * medida)


def exp_mms(outdir: Path, quick: bool) -> dict:
    print("\n[4/6] Orden formal por soluciones manufacturadas (MMS)")
    res_list = MMS_RES_QUICK if quick else MMS_RES_FULL
    casos = {
        "1d_none": dict(limiter="none", order=2, dos_d=False),
        "2d_none": dict(limiter="none", order=2, dos_d=True),
        "2d_mc": dict(limiter="mc", order=2, dos_d=True),
        "2d_minmod": dict(limiter="minmod", order=2, dos_d=True),
        "2d_order1": dict(limiter="minmod", order=1, dos_d=True),
    }
    out = {"resolutions": list(res_list)}
    curvas = {}
    for nombre, kw in casos.items():
        e = np.array([_mms_error(n, **kw) for n in res_list])
        _, pendiente = convergence_order(np.array(res_list), e)
        out[f"l1_errors_{nombre}"] = e.tolist()
        out[f"order_{nombre}"] = float(pendiente)
        curvas[nombre] = e
        print(f"      {nombre:12s} orden = {pendiente:.3f}")

    _figura_convergencia(outdir, res_list, curvas, out)
    return out


def _guardar_fig(fig, path) -> None:
    """
    PNG a 300 dpi + PDF vectorial, mismo nombre base.

    Las figuras que pasan por
    `utils.visualizer` usan su `_guardar()`, que hace exactamente esto; las dos
    figuras que se dibujan aqui (2 y 4) usan esta copia local para no obligar al
    runner a importar un helper privado.

    Sin `bbox_inches="tight"`: las figuras se crean con `layout="constrained"`.
    """
    from pathlib import Path as _P
    path = _P(path)
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    print(f"      [figura] {path}")
    print(f"      [figura] {path.with_suffix('.pdf')}")


def _figura_convergencia(outdir, res_list, curvas, out) -> None:
    """FIGURA 2: convergencia MMS (log-log) con pendientes de referencia."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("      [aviso] matplotlib ausente; se omite la figura 2")
        return
    n = np.array(res_list, dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 4.8), layout="constrained")
    etiquetas = {
        "1d_none": ("1D, unlimited (order 2 base)", "o-", "navy"),
        "2d_none": ("2D, unlimited (order 2 base)", "s-", "teal"),
        "2d_mc": ("2D, MC limiter", "^-", "darkorange"),
        "2d_minmod": ("2D, minmod limiter (production)", "v-", "crimson"),
        "2d_order1": ("2D, order 1 (negative control)", "d-", "0.5"),
    }
    for k, e in curvas.items():
        lab, mk, col = etiquetas[k]
        ax.loglog(n, e / e[0], mk, color=col, ms=4, lw=1.2,
                  label=f"{lab}  [p = {out[f'order_{k}']:.2f}]")
    ax.loglog(n, (n / n[0]) ** -1.0, "k:", lw=0.9, label="slope $-1$")
    ax.loglog(n, (n / n[0]) ** -2.0, "k--", lw=0.9, label="slope $-2$")
    ax.set_xlabel("Cells per direction $N$")
    ax.set_ylabel("Normalised $L_1$ error in $h$")
    ax.set_title("Formal order of accuracy (manufactured solution)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    _guardar_fig(fig, outdir / "fig2_mms_convergence.png")


# ===========================================================================
# 5. Barometro inverso estatico
# ===========================================================================
def exp_barometro_inverso() -> dict:
    print("\n[5/6] Barometro inverso estatico")
    L, nx, depth = IB["L"], IB["NX"], IB["DEPTH"]
    t_cruce = L / np.sqrt(TITAN.g * depth)
    dom = flat_channel(L, nx, depth)
    presion = StaticGaussianPressure(amplitude=IB["DP"], sigma=IB["SIGMA"],
                                     x0=0.5 * L, t_ramp=5.0 * t_cruce)
    res = run_to(dom, quiescent_state(dom),
                 ForcingBundle(pressure=presion,
                               friction=LinearFriction(r=IB["R"]),
                               coriolis_enabled=False),
                 SolverConfig(cfl=0.45, bc={"west": "reflective",
                                            "east": "reflective"}),
                 t_end=12.0 * t_cruce, sample_every=100)
    zeta = res.state.zeta
    p = presion.evaluate(dom.x, np.zeros_like(dom.x), res.state.t)
    i_c = nx // 2
    dz = zeta[i_c] - 0.5 * (zeta[:20].mean() + zeta[-20:].mean())
    dp = p[i_c] - 0.5 * (p[:20].mean() + p[-20:].mean())
    teo = inverse_barometer(dp)
    r = {"delta_p_effective": float(dp), "zeta_numeric": float(dz),
         "zeta_theory": float(teo),
         "rel_error": float(abs(dz - teo) / abs(teo))}
    print(f"      zeta = {dz:.6f} m vs {teo:.6f} m teorico "
          f"(error {r['rel_error']:.4%})")
    return r


# ===========================================================================
# 6. Barrido de resonancia  -> FIGURA 3 y FIGURA 4
# ===========================================================================
def _corrida_sweep(u, nx):
    dom = flat_channel(SW["L"], nx, SW["DEPTH"])
    presion = MovingGaussianPressure(amplitude=SW["DP"], sigma=SW["SIGMA"],
                                     speed=u, x0=SW["X0"])
    res = run_to(dom, quiescent_state(dom),
                 ForcingBundle(pressure=presion, coriolis_enabled=False),
                 SolverConfig(cfl=SW["CFL"], bc={"west": "transmissive",
                                                 "east": "transmissive"}),
                 t_end=SW["D"] / u, sample_every=10 ** 9)
    interior = (dom.x > SW["MARGEN"]) & (dom.x < SW["L"] - SW["MARGEN"])
    zeta, x = res.state.zeta[interior], dom.x[interior]
    return (amplification_from_field(zeta, SW["DP"]),
            bound_amplification(zeta, x, presion.center(res.state.t),
                                SW["DP"], 3.0, SW["SIGMA"]))


def exp_barrido(outdir: Path, quick: bool) -> dict:
    print("\n[6/6] Barrido de resonancia de Proudman")
    nx = SW_NX_QUICK if quick else SW_NX
    c = resonance_speed(SW["DEPTH"])
    n_speeds = 10 if quick else 19
    u = np.unique(np.concatenate([
        np.linspace(SWEEP_SPEED_RANGE[0], SWEEP_SPEED_RANGE[1], n_speeds), [c]]))

    A, Ab = [], []
    for uu in u:
        a, ab = _corrida_sweep(uu, nx)
        A.append(a)
        Ab.append(ab)
    A, Ab = np.array(A), np.array(Ab)

    peak_res = SW_PEAK_RES_QUICK if quick else SW_PEAK_RES_FULL
    picos = np.array([_corrida_sweep(c, n)[0] for n in peak_res])
    d = np.abs(np.diff(picos))
    razon = float(d[-2] / d[-1])
    richardson = float(picos[-1] + d[-1] / (razon - 1.0))

    plot_amplification_curve(
        u, A, SW["DEPTH"], outdir / "fig3_amplification_curve.png",
        delta_p=SW["DP"], bound_amplification=Ab,
        title=f"Proudman resonance sweep on Titan (h = {SW['DEPTH']:.0f} m)")
    _figura_pico(outdir, peak_res, picos, richardson)

    r = {"speeds": u.tolist(), "amplification": A.tolist(),
         "bound_amplification": Ab.tolist(),
         "froude": (u / c).tolist(),
         "peak_speed": float(u[int(np.argmax(A))]),
         "peak_amplification": float(A.max()),
         "resonance_speed": float(c),
         "nx": nx,
         "peak_resolutions": list(peak_res),
         "peak_by_resolution": picos.tolist(),
         "peak_convergence_ratio": razon,
         "peak_richardson": richardson,
         "production_bias": float((richardson - picos[1]) / richardson)}
    print(f"      maximo en U = {r['peak_speed']:.3f} m/s "
          f"(c = {c:.3f}), A = {r['peak_amplification']:.3f}")
    print(f"      pico extrapolado (Richardson) = {richardson:.4f}, "
          f"sesgo de produccion = {r['production_bias']:.2%}")
    return r


def _figura_pico(outdir, resoluciones, picos, richardson) -> None:
    """FIGURA 4: convergencia de malla de la altura del pico resonante."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    from matplotlib.ticker import NullLocator, ScalarFormatter

    dx = SW["L"] / np.array(resoluciones, dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 4.2), layout="constrained")
    ax.plot(dx, picos, "o-", color="crimson", ms=5, lw=1.3,
            label="Native FVM")
    ax.axhline(richardson, color="k", ls="--", lw=1.0,
               label=f"Richardson extrapolation = {richardson:.2f}")
    ax.set_xscale("log")
    ax.invert_xaxis()

    # Eje x: ticks EXACTAMENTE en los dx calculados. El localizador logaritmico
    # por defecto rotulaba 6x10^2, 4x10^2, 3x10^2, que no corresponden a ninguna
    # resolucion corrida: etiquetas inventadas en una figura de convergencia.
    ax.set_xticks(dx)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(axis="x", style="plain")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.margins(x=0.08)   # aire para que dx = 800 m no quede pegado al marco

    # Aire por encima de la linea de Richardson, que antes tachaba la leyenda.
    techo = max(float(np.max(picos)), float(richardson))
    suelo = float(np.min(picos))
    ax.set_ylim(suelo - 0.08 * (techo - suelo), techo + 0.18 * (techo - suelo))

    ax.set_xlabel("Grid spacing $\\Delta x$ [m]  (refining $\\rightarrow$)")
    ax.set_ylabel(r"Peak amplification $A(F=1)$")
    ax.set_title("Mesh convergence of the resonant peak")
    # Se retira la anotacion sobre el crecimiento del pico bajo refinamiento: el
    # caption del manuscrito ya lo enuncia en registro academico, y las
    # mayusculas enfaticas no van en una figura de revista. Quitarla libera
    # ademas el espacio inferior izquierdo, que estaba apretado.
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _guardar_fig(fig, outdir / "fig4_peak_mesh_convergence.png")


# ===========================================================================
# Consistencia 2D <-> 1D (numero, sin figura)
# ===========================================================================
def exp_consistencia_2d_1d() -> dict:
    L, nx, ny, depth = 3.0e5, 750, 8, 160.0
    dx = L / nx
    dp, sigma, x0 = 100.0, 1.0e4, 3.0e4
    c = resonance_speed(depth)
    u_storm = 0.95 * c
    t_end = 1.0e5 / u_storm
    dt = 6.0

    dom1 = flat_channel(L, nx, depth)
    s1 = NativeFVMSolver()
    s1.initialize(dom1, quiescent_state(dom1),
                  ForcingBundle(pressure=MovingGaussianPressure(
                      amplitude=dp, sigma=sigma, speed=u_storm, x0=x0),
                      coriolis_enabled=False),
                  SolverConfig(cfl=0.45, bc={"west": "transmissive",
                                             "east": "transmissive"}))
    dom2 = transverse_invariant_channel(L, ny * dx, nx, ny,
                                        np.full(nx, depth))
    s2 = NativeFVMSolver2D()
    s2.initialize(dom2, quiescent_state_2d(dom2),
                  ForcingBundle(pressure=MovingPressureFront2D(
                      amplitude=dp, sigma=sigma, speed=u_storm,
                      heading_deg=0.0, x0=x0, y0=0.0),
                      coriolis_enabled=False),
                  SolverConfig(cfl=0.45, bc={"west": "transmissive",
                                             "east": "transmissive",
                                             "south": "reflective",
                                             "north": "reflective"}))
    for s in (s1, s2):
        while s.state.t < t_end:
            s.step(min(dt, t_end - s.state.t))
    z1, z2 = s1.state.zeta, s2.state.zeta[0, :]
    err = float(np.max(np.abs(z2 - z1)) / np.max(np.abs(z1)))
    print(f"\n[extra] consistencia 2D<->1D con dt fijo: error relativo "
          f"= {err:.3e}")
    return {"rel_error_same_dt": err, "v_max_2d": float(np.max(np.abs(s2.state.v)))}


# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="resoluciones reducidas (para el test de reproduccion)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    np.random.seed(SEMILLA)   # el modelo es determinista; semilla por protocolo
    outdir = Path(args.outdir) if args.outdir else \
        (SNAPSHOT / ("outputs_quick" if args.quick else "outputs"))
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PAPER 1 - Reproduccion completa" + (" (modo rapido)" if args.quick
                                               else ""))
    print("=" * 78)
    print(f"Python {sys.version.split()[0]}  NumPy {np.__version__}")
    print(f"src/ = {SNAPSHOT / 'src'}")
    print(f"salida = {outdir}")
    print(f"g = {TITAN.g} m/s^2   rho = {DEFAULT_FLUID.rho} kg/m^3   "
          f"semilla = {SEMILLA}")

    numeros = {
        "meta": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "seed": SEMILLA,
            "quick": bool(args.quick),
            "g": TITAN.g,
            "rho": DEFAULT_FLUID.rho,
            "observed_front_speed": list(OBSERVED_FRONT_SPEED[:2]),
            "sweep_speed_range": list(SWEEP_SPEED_RANGE),
            "resonant_depth_band": list(resonant_depth_band()),
            "reference_depths": {k: v[0] for k, v in REFERENCE_DEPTHS.items()},
            "to_verify": audit_provenance(),
        },
        "gate1_well_balanced": exp_well_balanced(),
        "gate2_conservation": exp_conservacion(),
        "gate3_dambreak": exp_dambreak(outdir, args.quick),
        "gate4_mms": exp_mms(outdir, args.quick),
        "inverse_barometer": exp_barometro_inverso(),
        "gate5_sweep": exp_barrido(outdir, args.quick),
        "consistency_2d_1d": exp_consistencia_2d_1d(),
    }

    destino = outdir / "numbers.json"
    destino.write_text(json.dumps(numeros, indent=2, sort_keys=True),
                       encoding="utf-8")
    print(f"\n[numeros] {destino}")
    print("\nConstantes aun en TO_VERIFY (no publicar sin verificarlas):")
    for item in audit_provenance():
        print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
