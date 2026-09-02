"""
Bootstrapping CKKS con OpenFHE.

Demuestra cómo el bootstrapping restaura los niveles multiplicativos
de un cifrotexto agotado, permitiendo continuar operando.
Es la solución al «muro del ruido» que vimos con TenSEAL en quick_start.py.

Requisitos: pip install openfhe
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from i18n import t  # noqa: E402

from openfhe import (
    CCParamsCKKSRNS,
    FHECKKSRNS,
    GenCryptoContext,
    PKESchemeFeature,
    SecretKeyDist,
    SecurityLevel,
)


_CH = "cifrado-homomorfico"


def crear_contexto_bootstrap(niveles_utiles=2, num_slots=4):
    """Crea un contexto CKKS con bootstrapping habilitado.

    Args:
        niveles_utiles: multiplicaciones disponibles entre bootstraps.
        num_slots: número de valores por cifrotexto.

    Returns:
        (cc, keys, profundidad): contexto criptográfico, claves y
        profundidad total configurada.
    """
    params = CCParamsCKKSRNS()

    secret_key_dist = SecretKeyDist.UNIFORM_TERNARY
    params.SetSecretKeyDist(secret_key_dist)
    # Usamos un anillo pequeño (N = 4096) para que la demo sea rápida.
    # HEStd_NotSet desactiva la validación de nivel de seguridad,
    # necesaria porque N = 4096 no cumple HEStd_128_classic.
    params.SetSecurityLevel(SecurityLevel.HEStd_NotSet)
    params.SetRingDim(1 << 12)

    # El bootstrapping necesita niveles extra para evaluar
    # el circuito de descifrado homomórficamente.
    # level_budget = [niveles_colapsando, niveles_elevando]
    level_budget = [2, 2]

    profundidad = niveles_utiles + FHECKKSRNS.GetBootstrapDepth(
        level_budget, secret_key_dist
    )

    params.SetMultiplicativeDepth(profundidad)
    params.SetScalingModSize(48)
    params.SetFirstModSize(60)
    params.SetBatchSize(num_slots)

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)
    cc.Enable(PKESchemeFeature.FHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalBootstrapSetup(level_budget, [0, 0], num_slots)
    cc.EvalBootstrapKeyGen(keys.secretKey, num_slots)

    return cc, keys, profundidad


def bootstrapping_demo():
    """Replica la Receta 4 (muro del ruido) y lo supera con bootstrapping."""
    cc, keys, profundidad = crear_contexto_bootstrap(
        niveles_utiles=2, num_slots=4
    )

    datos = [3.0, 7.0, 2.0, 5.0]
    ptxt = cc.MakeCKKSPackedPlaintext(datos)
    ctxt = cc.Encrypt(keys.publicKey, ptxt)

    # --- x² (consume 1 nivel) ---
    ctxt = cc.EvalMult(ctxt, ctxt)

    # --- x⁴ (consume otro nivel — en TenSEAL esto fallaba) ---
    ctxt = cc.EvalMult(ctxt, ctxt)

    # Niveles agotados. En Receta 4: «Error: scale out of bounds»
    # Aquí: bootstrapping al rescate.
    t0 = time.perf_counter()
    ctxt = cc.EvalBootstrap(ctxt)
    t_bootstrap = time.perf_counter() - t0

    # --- x⁸ (¡ahora funciona!) ---
    ctxt = cc.EvalMult(ctxt, ctxt)

    # --- Descifrar y verificar ---
    result_ptxt = cc.Decrypt(ctxt, keys.secretKey)
    result_ptxt.SetLength(len(datos))
    valores = [v.real for v in result_ptxt.GetCKKSPackedValue()]

    esperado = [x**8 for x in datos]
    error_max = max(abs(v - e) for v, e in zip(valores, esperado))

    print(t(_CH, "resultado_fhe_boot").format(valores=[f'{v:.1f}' for v in valores]))
    print(t(_CH, "esperado_boot").format(valores=[f'{v:.1f}' for v in esperado]))
    print(t(_CH, "error_max_boot").format(valor=error_max))
    print(t(_CH, "tiempo_bootstrap").format(valor=t_bootstrap))

    return valores, esperado, t_bootstrap


if __name__ == "__main__":
    bootstrapping_demo()
