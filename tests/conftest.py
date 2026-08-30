"""Configuración de pytest para tests del capítulo cifrado-homomorfico."""

import os
import sys
from pathlib import Path

# Directorio con los scripts del capítulo (code/cifrado-homomorfico/)
CODE_DIR = str(Path(__file__).resolve().parent.parent.parent / "code" / "cifrado-homomorfico")
sys.path.insert(0, CODE_DIR)
