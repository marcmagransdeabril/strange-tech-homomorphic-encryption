"""
Cifrado homomórfico: ejemplo mínimo con TenSEAL.

Cifra una lista de salarios, calcula la media sobre los datos cifrados,
y descifra el resultado. En ningún momento el cálculo accede a los datos
en claro. Compara rendimiento entre FHE y cálculo en claro.

Requisitos: pip install tenseal
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from i18n import t  # noqa: E402

import tenseal as ts

_CH = "cifrado-homomorfico"


def crear_contexto():
    """Crea un contexto FHE con parámetros del esquema CKKS."""
    contexto = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    contexto.global_scale = 2**40
    contexto.generate_galois_keys()
    return contexto


def media_fhe(salarios, contexto):
    """Calcula la media sobre datos cifrados. Devuelve (resultado, tiempos)."""
    n = len(salarios)

    t0 = time.perf_counter()
    salarios_cifrados = ts.ckks_vector(contexto, salarios)
    t_cifrado = time.perf_counter() - t0

    t0 = time.perf_counter()
    suma_cifrada = salarios_cifrados.sum()
    media_cifrada = suma_cifrada * (1.0 / n)
    t_computo = time.perf_counter() - t0

    t0 = time.perf_counter()
    resultado = media_cifrada.decrypt()[0]
    t_descifrado = time.perf_counter() - t0

    tiempos = {
        "cifrado": t_cifrado,
        "computo": t_computo,
        "descifrado": t_descifrado,
        "total": t_cifrado + t_computo + t_descifrado,
    }
    return resultado, tiempos


def varianza_fhe(salarios, contexto):
    """Calcula la varianza poblacional sobre datos cifrados.

    Usa Var(X) = E[X²] - (E[X])². Para que la resta funcione en CKKS,
    ambos términos deben recorrer el mismo camino de operaciones
    (una multiplicación cifrado×cifrado, cero multiplicaciones por escalar).
    El truco: cifrar una copia pre-escalada por 1/n.
    """
    n = len(salarios)

    ct = ts.ckks_vector(contexto, salarios)
    # Pre-escalar en claro antes de cifrar: si dividimos después (ct * (1/n)),
    # la escala interna de CKKS diverge entre las dos ramas de la resta
    # y el resultado es basura.
    ct_n = ts.ckks_vector(contexto, [x / n for x in salarios])

    # E[X²] = sum(x_i · x_i/n): una mult cifrado×cifrado (nivel 1)
    media_cuadrados = (ct * ct_n).sum()

    # (E[X])² = (sum(x_i/n))²: una mult cifrado×cifrado (nivel 1)
    media = ct_n.sum()
    media_al_cuadrado = media * media

    # Ambos al mismo nivel y escala — la resta funciona
    varianza_cifrada = media_cuadrados - media_al_cuadrado

    return varianza_cifrada.decrypt()[0]


def regresion_fhe(features, pesos, contexto):
    """Predicción de regresión lineal sobre features cifrados.

    Los pesos del modelo (en claro) se multiplican por los features
    (cifrados) y se suman — un producto escalar homomórfico.
    El servidor nunca ve los datos del paciente.
    """
    features_cifrados = ts.ckks_vector(contexto, features)

    # Multiplicación elemento a elemento por constantes (barata, no consume nivel)
    producto = features_cifrados * pesos

    # Suma todas las posiciones (usa rotaciones + claves de Galois)
    prediccion_cifrada = producto.sum()

    return prediccion_cifrada.decrypt()[0]


def busqueda_fhe(indice, base_datos, contexto):
    """Recupera un registro por índice sin revelar cuál se consulta (PIR).

    El cliente cifra un vector one-hot con un 1 en la posición deseada.
    El servidor multiplica ese vector por la base de datos (en claro)
    y suma — obteniendo el valor seleccionado, cifrado.
    El servidor nunca sabe qué posición se consultó.
    """
    n = len(base_datos)
    # Vector one-hot: 1 en la posición consultada, 0 en el resto
    consulta = [1.0 if i == indice else 0.0 for i in range(n)]
    consulta_cifrada = ts.ckks_vector(contexto, consulta)

    # El servidor solo ve el cifrotexto — no sabe dónde está el 1
    resultado_cifrado = (consulta_cifrada * base_datos).sum()

    return resultado_cifrado.decrypt()[0]


def muro_del_ruido():
    """Demuestra qué ocurre al exceder el presupuesto de niveles.

    Crea un contexto con un solo nivel multiplicativo e intenta
    dos multiplicaciones encadenadas. La segunda falla.
    """
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 60],  # solo 1 nivel
    )
    ctx.global_scale = 2**40
    ctx.generate_galois_keys()

    datos = ts.ckks_vector(ctx, [3.0, 7.0, 2.0])

    # Primera multiplicación: OK (consume el único nivel)
    resultado = datos * datos
    valores = resultado.decrypt()
    print(t(_CH, "tras_1_mult").format(valores=[f'{v:.1f}' for v in valores]))

    # Segunda multiplicación: falla — no quedan niveles
    try:
        resultado = resultado * resultado
        print(t(_CH, "error_segunda_mult"))
        return False
    except Exception as e:
        print(t(_CH, "muro_alcanzado").format(error=f"{type(e).__name__}: {e}"))
        return True


def main():
    contexto = crear_contexto()
    salarios = [3200, 4100, 2800, 5500, 3900]
    n = len(salarios)

    # --- Receta 1: Media cifrada ---
    t0 = time.perf_counter()
    media_claro = sum(salarios) / n
    t_claro = time.perf_counter() - t0

    resultado, tiempos = media_fhe(salarios, contexto)

    print(t(_CH, "receta1_header"))
    print(t(_CH, "media_fhe_label").format(valor=resultado))
    print(t(_CH, "media_claro_label").format(valor=media_claro))
    print(t(_CH, "diferencia_label").format(valor=abs(resultado - media_claro)))
    print(t(_CH, "slowdown_label").format(valor=tiempos['total'] / max(t_claro, 1e-9)))
    print()

    # --- Receta 2: Varianza cifrada ---
    var_claro = sum((x - media_claro) ** 2 for x in salarios) / n
    var_fhe = varianza_fhe(salarios, contexto)

    print(t(_CH, "receta2_header"))
    print(t(_CH, "varianza_fhe_label").format(valor=var_fhe))
    print(t(_CH, "varianza_claro_label").format(valor=var_claro))
    print(t(_CH, "diferencia_var_label").format(valor=abs(var_fhe - var_claro)))
    print()

    # --- Receta 3: Regresión lineal cifrada ---
    features = [1.2, 0.7, 3.1, 0.4, 2.8]  # datos del paciente (cifrados)
    pesos = [0.5, -1.2, 0.8, 0.3, -0.6]   # modelo entrenado (en claro)
    pred_claro = sum(f * w for f, w in zip(features, pesos))
    pred_fhe = regresion_fhe(features, pesos, contexto)

    print(t(_CH, "receta3_header"))
    print(t(_CH, "pred_fhe_label").format(valor=pred_fhe))
    print(t(_CH, "pred_claro_label").format(valor=pred_claro))
    print(t(_CH, "diferencia_pred_label").format(valor=abs(pred_fhe - pred_claro)))
    print()

    # --- Receta 4: El muro del ruido ---
    print(t(_CH, "receta4_header"))
    muro_del_ruido()
    print()

    # --- Receta 6: Búsqueda cifrada (PIR) ---
    base_datos = [85.2, 62.0, 91.7, 45.3, 78.9, 33.1, 70.4, 56.8]
    indice_consulta = 3
    valor_claro = base_datos[indice_consulta]
    valor_fhe = busqueda_fhe(indice_consulta, base_datos, contexto)

    print(t(_CH, "receta6_header"))
    print(t(_CH, "indice_label").format(valor=indice_consulta))
    print(t(_CH, "valor_fhe_label").format(valor=valor_fhe))
    print(t(_CH, "valor_claro_label").format(valor=valor_claro))
    print(t(_CH, "diferencia_pir_label").format(valor=abs(valor_fhe - valor_claro)))
    print(t(_CH, "registros_label").format(valor=len(base_datos)))
    print(t(_CH, "servidor_msg"))


if __name__ == "__main__":
    main()
