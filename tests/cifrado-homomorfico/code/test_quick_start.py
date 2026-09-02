"""Tests para el ejemplo de cifrado homomórfico."""

import subprocess
import sys
import os

import pytest

CODE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "code", "cifrado-homomorfico")


def test_script_ejecuta_o_falla_por_dependencia():
    """El script debe ejecutarse sin errores si tenseal está instalado,
    o fallar limpiamente con ImportError si no lo está."""
    result = subprocess.run(
        [sys.executable, os.path.join(CODE_DIR, "quick_start.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        # Aceptable: fallo por falta de tenseal
        assert "No module named" in result.stderr or "ModuleNotFoundError" in result.stderr, (
            f"Fallo inesperado (exit {result.returncode}):\n{result.stderr}"
        )
        pytest.skip("tenseal no instalado")
    if os.environ.get("BOOK_LANG") == "en":
        assert "Recipe 1: Encrypted mean" in result.stdout
        assert "Recipe 2: Encrypted variance" in result.stdout
        assert "Recipe 3: Encrypted linear regression" in result.stdout
        assert "Recipe 4: The noise wall" in result.stdout
        assert "Recipe 6: Encrypted search" in result.stdout
    else:
        assert "Receta 1: Media cifrada" in result.stdout
        assert "Receta 2: Varianza cifrada" in result.stdout
        assert "Receta 3: Regresión lineal cifrada" in result.stdout
        assert "Receta 4: El muro del ruido" in result.stdout
        assert "Receta 6: Búsqueda cifrada" in result.stdout


@pytest.fixture(scope="module")
def tenseal_ctx():
    """Crea un contexto FHE si tenseal está disponible."""
    ts = pytest.importorskip("tenseal", reason="tenseal no instalado")
    sys.path.insert(0, CODE_DIR)
    from quick_start import crear_contexto
    return crear_contexto()


def test_media_fhe_coincide_con_claro(tenseal_ctx):
    """La media calculada sobre datos cifrados debe coincidir con la de claro."""
    sys.path.insert(0, CODE_DIR)
    from quick_start import media_fhe

    salarios = [3200, 4100, 2800, 5500, 3900]
    media_claro = sum(salarios) / len(salarios)

    resultado, tiempos = media_fhe(salarios, tenseal_ctx)

    assert abs(resultado - media_claro) < 0.01, (
        f"FHE={resultado:.4f} vs claro={media_claro:.4f}"
    )
    assert tiempos["total"] > 0


def test_media_fhe_un_solo_elemento(tenseal_ctx):
    """FHE con un solo dato debe devolver ese dato."""
    sys.path.insert(0, CODE_DIR)
    from quick_start import media_fhe

    resultado, _ = media_fhe([42.0], tenseal_ctx)
    assert abs(resultado - 42.0) < 0.01


def test_varianza_fhe_coincide_con_claro(tenseal_ctx):
    """La varianza calculada sobre datos cifrados debe coincidir con la de claro."""
    sys.path.insert(0, CODE_DIR)
    from quick_start import varianza_fhe

    salarios = [3200, 4100, 2800, 5500, 3900]
    media = sum(salarios) / len(salarios)
    var_claro = sum((x - media) ** 2 for x in salarios) / len(salarios)

    var_fhe = varianza_fhe(salarios, tenseal_ctx)

    assert abs(var_fhe - var_claro) < 1.0, (
        f"FHE={var_fhe:.4f} vs claro={var_claro:.4f}"
    )


def test_regresion_fhe_coincide_con_claro(tenseal_ctx):
    """El producto escalar cifrado debe coincidir con el cálculo en claro."""
    sys.path.insert(0, CODE_DIR)
    from quick_start import regresion_fhe

    features = [1.2, 0.7, 3.1, 0.4, 2.8]
    pesos = [0.5, -1.2, 0.8, 0.3, -0.6]
    pred_claro = sum(f * w for f, w in zip(features, pesos))

    pred_fhe = regresion_fhe(features, pesos, tenseal_ctx)

    assert abs(pred_fhe - pred_claro) < 0.01, (
        f"FHE={pred_fhe:.4f} vs claro={pred_claro:.4f}"
    )


def test_muro_del_ruido_falla_gracefully():
    """El muro del ruido debe detectar el fallo y devolver True."""
    ts = pytest.importorskip("tenseal", reason="tenseal no instalado")
    sys.path.insert(0, CODE_DIR)
    from quick_start import muro_del_ruido

    assert muro_del_ruido() is True


def test_busqueda_fhe_coincide_con_claro(tenseal_ctx):
    """La búsqueda cifrada debe devolver el valor correcto del índice consultado."""
    sys.path.insert(0, CODE_DIR)
    from quick_start import busqueda_fhe

    base_datos = [85.2, 62.0, 91.7, 45.3, 78.9, 33.1, 70.4, 56.8]
    indice = 3

    resultado = busqueda_fhe(indice, base_datos, tenseal_ctx)

    assert abs(resultado - base_datos[indice]) < 0.01, (
        f"FHE={resultado:.4f} vs claro={base_datos[indice]:.4f}"
    )


def test_busqueda_fhe_primer_y_ultimo(tenseal_ctx):
    """La búsqueda cifrada funciona en los extremos del vector."""
    sys.path.insert(0, CODE_DIR)
    from quick_start import busqueda_fhe

    base_datos = [10.0, 20.0, 30.0, 40.0, 50.0]

    primero = busqueda_fhe(0, base_datos, tenseal_ctx)
    assert abs(primero - 10.0) < 0.01

    ultimo = busqueda_fhe(4, base_datos, tenseal_ctx)
    assert abs(ultimo - 50.0) < 0.01
