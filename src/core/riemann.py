"""
core/riemann.py
===============
Solvers de Riemann aproximados para las Ecuaciones de Agua Poco Profunda 1D.

Sistema resuelto (variables conservadas, internas al motor; ver ADR-001):
    q = (h, hu, hv)
    F(q) = (hu, hu^2/h + g h^2/2, hu*hv/h)

`hv` es momento transversal transportado pasivamente por la onda de contacto:
en 1D solo importa cuando Coriolis esta activo.

SUPUESTOS NUMERICOS EXPLICITOS
------------------------------
* Estimador de velocidades de onda: Einfeldt-Batten (media de Roe + estados
  laterales), con la correccion de estado seco de Toro para h -> 0.
* HLLC de tres ondas: S_L, S_* (contacto) y S_R.
* Consistencia con el reposo: si (h_L,u_L) == (h_R,u_R) y u == 0, el flujo
  devuelto es EXACTAMENTE (0, g h^2/2, 0) en aritmetica de punto flotante.
  Esa propiedad es la que sostiene la compuerta 1 (lago en reposo).

GANCHO DE ACELERACION: las funciones son puramente vectoriales sobre arrays de
NumPy; son candidatas directas a @numba.njit o jax.jit sin cambio de firma.
No se optimiza aqui de forma prematura.
"""

from __future__ import annotations
import numpy as np

__all__ = ["wave_speeds", "hllc_flux", "physical_flux", "hydrostatic_pressure"]


def hydrostatic_pressure(h: np.ndarray, g: float) -> np.ndarray:
    """
    Termino de presion hidrostatica del flujo de momento: g h^2 / 2.

    Se aisla en una funcion para que el motor pueda RESTARLO del flujo numerico
    con cancelacion EXACTA en punto flotante (misma secuencia de operaciones).
    De ahi depende el equilibrio de reposo a cero bit a bit.
    """
    return 0.5 * g * h * h


def _velocity(h: np.ndarray, hu: np.ndarray, dry_tol: float) -> np.ndarray:
    """u = hu/h desingularizado: cero en celdas por debajo del umbral seco."""
    return np.where(h > dry_tol, hu / np.where(h > dry_tol, h, 1.0), 0.0)


def physical_flux(h: np.ndarray, u: np.ndarray, v: np.ndarray,
                  g: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flujo fisico exacto F(q) en la direccion x."""
    hu = h * u
    return hu, hu * u + hydrostatic_pressure(h, g), hu * v


def wave_speeds(hL: np.ndarray, uL: np.ndarray,
                hR: np.ndarray, uR: np.ndarray,
                g: float, dry_tol: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimacion Einfeldt de las velocidades de onda extremas (S_L, S_R).

    Para estados identicos con u=0 devuelve exactamente (-sqrt(g h), +sqrt(g h)),
    condicion necesaria para el equilibrio de reposo exacto.
    """
    cL = np.sqrt(g * np.maximum(hL, 0.0))
    cR = np.sqrt(g * np.maximum(hR, 0.0))

    # Media de Roe (ponderada por sqrt(h)) y celeridad de Roe.
    sq_hL = np.sqrt(np.maximum(hL, 0.0))
    sq_hR = np.sqrt(np.maximum(hR, 0.0))
    denom = sq_hL + sq_hR
    safe = denom > 0.0
    u_roe = np.where(safe, (sq_hL * uL + sq_hR * uR) / np.where(safe, denom, 1.0), 0.0)
    c_roe = np.sqrt(0.5 * g * (np.maximum(hL, 0.0) + np.maximum(hR, 0.0)))

    sL = np.minimum(uL - cL, u_roe - c_roe)
    sR = np.maximum(uR + cR, u_roe + c_roe)

    # Correccion de estado seco (Toro): frente de rarefaccion sobre fondo seco.
    dryL = hL <= dry_tol
    dryR = hR <= dry_tol
    sL = np.where(dryL, uR - 2.0 * cR, sL)
    sR = np.where(dryL, uR + cR, sR)
    sL = np.where(dryR, uL - cL, sL)
    sR = np.where(dryR, uL + 2.0 * cL, sR)

    # Ambos lados secos: sin ondas.
    both_dry = dryL & dryR
    sL = np.where(both_dry, 0.0, sL)
    sR = np.where(both_dry, 0.0, sR)
    return sL, sR


def hllc_flux(hL: np.ndarray, uL: np.ndarray, vL: np.ndarray,
              hR: np.ndarray, uR: np.ndarray, vR: np.ndarray,
              g: float, dry_tol: float = 1e-12
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Flujo numerico HLLC en una interfaz.

    Devuelve (F_h, F_hu, F_hv, s_max) donde s_max = max(|S_L|,|S_R|) se reutiliza
    para el CFL dinamico.
    """
    sL, sR = wave_speeds(hL, uL, hR, uR, g, dry_tol)

    FhL, FhuL, FhvL = physical_flux(hL, uL, vL, g)
    FhR, FhuR, FhvR = physical_flux(hR, uR, vR, g)

    # Velocidad de la onda de contacto S_* (formula de Toro para SWE).
    # Para estados identicos se reduce exactamente a u.
    num = sL * hR * (uR - sR) - sR * hL * (uL - sL)
    den = hR * (uR - sR) - hL * (uL - sL)
    s_star = np.where(np.abs(den) > 0.0, num / np.where(np.abs(den) > 0.0, den, 1.0), 0.0)

    # Estados intermedios.
    def _star(h, u, v, s):
        fac = np.where(np.abs(s - s_star) > 0.0,
                       (s - u) / np.where(np.abs(s - s_star) > 0.0, s - s_star, 1.0),
                       1.0)
        h_s = h * fac
        return h_s, h_s * s_star, h_s * v

    hsL, husL, hvsL = _star(hL, uL, vL, sL)
    hsR, husR, hvsR = _star(hR, uR, vR, sR)

    Fh = np.where(sL >= 0.0, FhL,
         np.where(s_star >= 0.0, FhL + sL * (hsL - hL),
         np.where(sR > 0.0, FhR + sR * (hsR - hR), FhR)))
    Fhu = np.where(sL >= 0.0, FhuL,
          np.where(s_star >= 0.0, FhuL + sL * (husL - hL * uL),
          np.where(sR > 0.0, FhuR + sR * (husR - hR * uR), FhuR)))
    Fhv = np.where(sL >= 0.0, FhvL,
          np.where(s_star >= 0.0, FhvL + sL * (hvsL - hL * vL),
          np.where(sR > 0.0, FhvR + sR * (hvsR - hR * vR), FhvR)))

    s_max = np.maximum(np.abs(sL), np.abs(sR))
    return Fh, Fhu, Fhv, s_max
