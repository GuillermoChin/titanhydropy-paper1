"""
conftest.py — Paper 1 (snapshot inmutable)
==========================================
RE-ENRAIZADO DE IMPORTS. Inserta `src/` en la POSICION 0 de sys.path, de modo
que `from core... import ...` resuelva SIEMPRE contra la copia congelada de este
snapshot y nunca contra el codigo vivo del repositorio, aunque el snapshot se
ejecute desde dentro del repo.

tests/test_reproduction.py comprueba explicitamente de que ruta se cargaron los
modulos: si alguna vez se colara el codigo vivo, el test falla.
"""

from __future__ import annotations

import sys
from pathlib import Path

SNAPSHOT = Path(__file__).resolve().parent
SRC = SNAPSHOT / "src"

# Posicion 0: gana sobre cualquier otra entrada, incluida la raiz del repo.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
elif sys.path[0] != str(SRC):
    sys.path.remove(str(SRC))
    sys.path.insert(0, str(SRC))
