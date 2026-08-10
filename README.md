# Paper 1 — Code snapshot v2 (immutable)

Frozen, self-contained snapshot of the TitanHydroPy code that produced every
number and every figure of Paper 1.

> **This directory is immutable.** It is never modified by later sprints. The
> live code in the repository root continues to evolve; this snapshot does not.
> `SHA256SUMS` and `tests/test_reproduction.py` enforce that.

## Relationship to v1 — read this before comparing the two

This is a **reissue**. The previous snapshot lives on, untouched, in
`Paper_1_Code/`; it was neither modified nor renamed, because an immutable
snapshot that gets moved is no longer immutable. Both versions coexist.

**The physics, the solver and every published number are unchanged.** The full
reproduction of v2 is bit-for-bit identical to v1 across all 108 reported
quantities. What changed is documentation, one citation, and figure
presentation:

| | v1 | v2 |
|---|---|---|
| Dam-break L1 order (README text) | 0.985 | **0.9970** |
| Peak convergence ratio (README text) | ≈2.45 | **2.336** |
| Effective order of peak convergence | ≈1.3 | **1.16–1.22** |
| Charnay et al. 2015 pages | 344–348 | **362–366** |
| Figures | PNG at 150 dpi | PNG at 300 dpi **+ vector PDF** |

### Why the v1 numbers were wrong, and what that does *not* affect

The v1 README reported the mesh convergence of the resonant peak as a **ratio of
successive relative changes** (10.74 % / 4.35 % and 4.35 % / 1.78 %, giving 2.47
and 2.44, quoted as "≈2.45"). Richardson extrapolation requires the ratio of
**absolute differences**. Relative changes are normalised by different
denominators, so their quotient is not a convergence ratio at all.

With the absolute differences — 1.0912, 0.4892, 0.2094 — the ratios are **2.231
and 2.336**, and the effective order is **1.16–1.22**, not ≈1.3.

**The computation was always correct; only the prose was wrong.**
`reproduce/run_all.py::exp_barrido` has always used `np.abs(np.diff(picos))`, and
both v1 and v2 store `peak_convergence_ratio = 2.336067` in
`expected_outputs/numbers.json`. Consequently `A∞ = 12.104` and the
production-resolution bias of **7.07 %** are unaffected and stand as published.

The v1 dam-break figure of 0.985 was simply **stale**: it matches no
configuration this snapshot ships. The full ladder (nx = 100…3200) gives 0.9970
and `--quick` (100…800) gives 0.9876.

**When the two versions disagree, `expected_outputs/numbers.json` is the source
of truth in both — not the README.**

## Run everything with one command

```bash
python reproduce/run_all.py
```

Regenerates all figures and `numbers.json` into `outputs/` (~15 min). No network
access, no external data, no manual steps.

```bash
python reproduce/run_all.py --quick
```

Reduced resolutions (~1 min), used by the reproduction test.

### Install

```bash
python -m pip install -r requirements.txt
```

Only NumPy is required to compute; matplotlib is needed for the figures and is
optional (the run degrades gracefully without it). Python ≥ 3.10.

### Verify the snapshot

```bash
python -m pytest tests/test_reproduction.py -q
```

Checks three things: that modules load from this snapshot's `src/` and not from
the live repository, that `SHA256SUMS` still matches, and that a quick
reproduction lands on the published numbers.

## Script → figure map

Every figure is written twice: `.png` at 300 dpi and `.pdf` as vector art, both
with the same basename. 

| Figure | Produced by | What it shows |
|---|---|---|
| `fig1_dambreak_profile.png` | `run_all.py::exp_dambreak` | 1D dam break at t = 100 s, numerical vs **exact** Riemann solution, nx = 400 |
| `fig2_mms_convergence.png` | `run_all.py::exp_mms` | Formal order of accuracy on a smooth manufactured solution, five configurations, with slope −1 and −2 references |
| `fig3_amplification_curve.png` | `run_all.py::exp_barrido` | **Main science figure.** Amplification `A = max|ζ|/|ζ_IB|` vs storm speed `U`, with steady linear theory `1/\|1−F²\|` and the observed front-speed band |
| `fig4_peak_mesh_convergence.png` | `run_all.py::exp_barrido` | Mesh convergence of the resonant peak height, with Richardson extrapolation |

All numbers land in `outputs/numbers.json`; the published reference copy is
`expected_outputs/numbers.json` (full) and `expected_outputs/quick/numbers.json`
(quick mode).

## Validation gates reproduced by this snapshot

Every gate is a first-class test, not an optional script.

| Gate | Result |
|---|---|
| 1. Lake at rest (well-balanced), 1D and 2D | `max\|ζ\| = max\|u\| = max\|v\| = 0.0` — **exact to the bit**, on variable bathymetry, orders 1 and 2, all limiters, all four boundary types, mixed corners, anisotropic mesh, Coriolis on |
| 2. Conservation, closed domain | mass drift < 1×10⁻¹², energy strictly non-increasing |
| 3. Dam break vs exact Riemann | L1 order **0.9970** — capped at ~1 by the shock, as theory requires |
| 4. 2D ↔ 1D consistency | machine precision with matched dt (9×10⁻¹³); rotating the problem 90° changes the answer by **exactly 0.0** |
| 5. Formal order (MMS, smooth) | base scheme **2.007** (1D) / **1.982** (2D); MC limiter 2.029; minmod (production) 1.805; `order=1` negative control 0.958 |
| 6. Static inverse barometer | ζ = −0.164316 m vs −0.164350 m theoretical → **0.021 %** error |
| 7. Resonance sweep | peak exactly at `U = c = √(gh)`; peak **converges** under refinement, ratio of absolute differences **2.231 and 2.336** (effective order **1.16–1.22**), `A∞ = 12.104` |
| 8. Titan physical checks | all Level-C checks pass |

## What the science figure says

At `h = 160 m` (Ligeia Mare's maximum depth), `c = √(gh) = 14.71 m/s`. The
amplification peaks sharply there and reaches `A ≈ 11.2` at production
resolution (`12.10` extrapolated to zero mesh spacing; production underestimates
by **7.07 %**, reported rather than hidden).

**But the resonant speed lies outside the observed front-speed range.** With
`OBSERVED_FRONT_SPEED = 2–10 m/s` (Charnay et al. 2015) the resonant depth band
is `h = U²/g ∈ [3, 74] m`, below the maximum depth of every catalogued sea.
Within the observed range and at 160 m the amplification only reaches 1.86.

The consequence — resonance in Titan's seas is a phenomenon of the
**intermediate-depth flanks**, not the deep basin centres — is recorded as
**H-002** in the project's `Hallazgos.md`. **It rests entirely on a `TO_VERIFY`
constant** (Charnay reports gust-front *wind*, at equatorial latitudes) and is
not publishable until that primary source is closed. See `PROVENANCE.md`.

## Layout

```
src/                 frozen packages (constants, core, domain, forcing,
                     physics, titan_io, utils, validation)
reproduce/run_all.py one-command regeneration, fixed seed
expected_outputs/    published figures (PNG + PDF) and numbers.json
                     (+ quick/ reference)
tests/               reproduction test + the full gate suite as shipped
conftest.py          import re-rooting: src/ at sys.path position 0
requirements.txt     pip freeze of the reference environment
PROVENANCE.md        provenance of every physical value
SHA256SUMS           integrity of every file above
```

### On import re-rooting

Packages live under `src/`, and every entry point inserts `src/` at **position 0**
of `sys.path` before importing anything. `from core... import ...` therefore
always resolves against this frozen copy, even when the snapshot is executed
from inside the live repository. `test_los_modulos_se_cargan_del_snapshot_no_del_codigo_vivo`
asserts this by inspecting `module.__file__`.

### Determinism

The model uses no random numbers; it is fully deterministic. A seed
(`20250101`) is fixed by protocol anyway. Reproduction is bit-for-bit on the
same machine and NumPy build; across machines, floating-point operation order
may differ, so the reproduction test allows 1×10⁻⁹ relative — except for the
quantities that must be **exactly zero** (well-balanced), which are checked as
exact zero.

## License

MIT — see `LICENSE`. Citation metadata in `CITATION.cff`.
