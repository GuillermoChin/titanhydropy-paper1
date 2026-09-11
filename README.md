# Paper 1 — Code snapshot v1.0.1 (immutable)

Frozen, self-contained snapshot of the TitanHydroPy code that produced every
number and every figure of Paper 1.

> **This directory is immutable.** The development code it was frozen from
> continues to evolve; this archive does not. `SHA256SUMS` and
> `tests/test_reproduction.py` enforce that.

## About this release

This archive accompanies the manuscript and contains everything needed to
regenerate its figures and numbers from scratch.

It **supersedes an earlier issue tagged `v1.0.0`**. Corrections to a released
archive are made by **reissuing** it under a new tag, never by editing a release
in place: an archive that can be edited after the fact is no longer a record of
anything. Earlier issues remain available and unmodified.

**The physics, the solver and every reported number are unchanged across
issues.** Reproduction is bit-for-bit identical over all 158 reported quantities
and over all eight figure files.

**`expected_outputs/numbers.json` is the source of truth for every number in this
archive — not this README.**

### Two levels of reproducibility, and which one holds where

The **numbers** and the **images** reproduce anywhere: any machine, any supported
platform, any compatible version of the dependencies. That is the guarantee the
science rests on, and `expected_outputs/numbers.json` is where it is recorded.

The **bytes** of the figure files reproduce inside the pinned environment of
`requirements.txt`. Figure files are containers: a PNG is an encoded image and a
PDF is an encoded document, and the encoders live in the dependency chain. A
different version of an encoder can store the very same image as a different
byte stream, which is why `requirements.txt` pins the packages that determine
those bytes — including transitive ones nobody imports directly.

So if `sha256sum -c` reports a figure mismatch after you regenerate, check your
environment against `requirements.txt` before concluding that anything changed:
decode both files and compare the images. Identical images with different bytes
mean a different encoder, not a different result.

### A note on the dam-break L1 order

`l1_order` is the slope of a log–log fit over whatever resolutions it is given,
so it is a property of the **mesh ladder** as much as of the scheme. Over this
snapshot's full ladder (nx = 100…3200) it is **0.9970**; over `--quick`
(nx = 100…800) it is **0.9876**. A figure quoted without its ladder is
ambiguous, so this archive always states it.

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

`requirements.txt` also pins the transitive packages that determine the **bytes**
of the figure files. Install from it if you intend to check the figure checksums;
for the numbers and the images alone, any compatible environment will do.

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
| 3. Dam break vs exact Riemann | L1 order **0.9970** over the nx = 100…3200 ladder — capped at ~1 by the shock, as theory requires |
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

The consequence is that resonance in Titan's seas is a phenomenon of the
**intermediate-depth flanks**, not of the deep basin centres. **It rests entirely
on a `TO_VERIFY` constant** — Charnay reports gust-front *wind*, measured at
equatorial latitudes — and is not publishable until that primary source is
closed. See `PROVENANCE.md`.

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
