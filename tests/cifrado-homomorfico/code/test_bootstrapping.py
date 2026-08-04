"""Tests para el ejemplo de bootstrapping CKKS con OpenFHE."""

import subprocess
import sys
import os

import pytest

CODE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "code", "cifrado-homomorfico")


def test_script_ejecuta_o_falla_por_dependencia():
    """El script debe ejecutarse sin errores si openfhe está instalado,
    o fallar limpiamente con ImportError si no lo está."""
    result = subprocess.run(
        [sys.executable, os.path.join(CODE_DIR, "bootstrapping.py")],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        assert (
            "No module named" in result.stderr
            or "ModuleNotFoundError" in result.stderr
        ), f"Fallo inesperado (exit {result.returncode}):\n{result.stderr}"
        pytest.skip("openfhe no instalado")
    assert "Resultado FHE" in result.stdout
    assert "Tiempo bootstrap" in result.stdout


def test_bootstrapping_resultado_correcto():
    """El bootstrapping debe producir x⁸ con error acotado."""
    fhe = pytest.importorskip("openfhe", reason="openfhe no instalado")
    sys.path.insert(0, CODE_DIR)
    from bootstrapping import bootstrapping_demo

    valores, esperado, t_bootstrap = bootstrapping_demo()

    for v, e in zip(valores, esperado):
        assert abs(v - e) < 1.0, f"FHE={v:.4f} vs esperado={e:.4f}"
    assert t_bootstrap > 0
