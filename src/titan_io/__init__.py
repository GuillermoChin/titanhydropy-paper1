"""
Entrada/salida de TitanHydroPy.

NOMBRE DEL PAQUETE: el arbol acordado lo llamaba `io/`. Un paquete `io` en la
raiz de sys.path ensombrece el modulo `io` de la biblioteca estandar y rompe
el import de numpy. Ver docs/DECISIONS.md (ADR-002).
"""

from .writers import (  # noqa: F401
    TimeSeriesRecorder, save_state, load_state, save_fields, load_fields,
)
