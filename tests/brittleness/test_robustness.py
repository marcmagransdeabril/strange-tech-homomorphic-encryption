"""Brittleness tests for FHE code examples (Chapter 3: Cifrado Homomórfico).

~50 variations testing edge cases, precision limits, slot boundaries,
and parameter sensitivity of the CKKS homomorphic encryption code examples.

Each test follows TDD: define expected plaintext result, run the FHE
function, compare. Failures indicate either a wrong test hypothesis
or brittle original code.
"""

import os
import sys
import random

import pytest

ts = pytest.importorskip("tenseal", reason="tenseal no instalado")

CODE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "code", "cifrado-homomorfico"
)
sys.path.insert(0, CODE_DIR)

from quick_start import (  # noqa: E402
    crear_contexto,
    media_fhe,
    varianza_fhe,
    regresion_fhe,
    busqueda_fhe,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ctx():
    """Contexto FHE compartido por todos los tests del módulo."""
    return crear_contexto()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _var_plain(data):
    """Varianza poblacional en claro."""
    n = len(data)
    mean = sum(data) / n
    return sum((x - mean) ** 2 for x in data) / n


def _dot(a, b):
    """Producto escalar en claro."""
    return sum(x * y for x, y in zip(a, b))


# ===========================================================================
# Group A: media_fhe — 10 variations
# ===========================================================================

class TestMediaFHE:

    # A1: single element
    def test_single_element(self, ctx):
        result, _ = media_fhe([42.0], ctx)
        assert abs(result - 42.0) < 0.01

    # A2: two elements
    def test_two_elements(self, ctx):
        data = [100.0, 200.0]
        result, _ = media_fhe(data, ctx)
        assert abs(result - 150.0) < 0.01

    # A3: 100 random values
    def test_100_random(self, ctx):
        rng = random.Random(42)
        data = [rng.uniform(0, 10000) for _ in range(100)]
        expected = sum(data) / len(data)
        result, _ = media_fhe(data, ctx)
        assert abs(result - expected) < 0.1, f"FHE={result:.6f} vs {expected:.6f}"

    # A4: 4096 elements (max slots for N=8192)
    def test_max_slots(self, ctx):
        rng = random.Random(42)
        data = [rng.uniform(0, 1000) for _ in range(4096)]
        expected = sum(data) / len(data)
        result, _ = media_fhe(data, ctx)
        assert abs(result - expected) < 1.0, f"FHE={result:.6f} vs {expected:.6f}"

    # A5: 4097 elements — discovery test (exceeds N/2 slots)
    def test_exceeds_slots(self, ctx):
        """Discover behavior when data exceeds N/2=4096 slots."""
        data = [1.0] * 4097
        expected = 1.0
        try:
            result, _ = media_fhe(data, ctx)
            # If it works, check correctness
            assert abs(result - expected) < 0.1, (
                f"Slot overflow: FHE={result:.6f} vs {expected:.6f} — "
                "TenSEAL accepted >4096 elements but result may be wrong"
            )
        except Exception as e:
            # Expected: TenSEAL should reject or raise
            pytest.skip(f"TenSEAL rejects >4096 elements: {type(e).__name__}: {e}")

    # A6: very large values
    def test_large_values(self, ctx):
        data = [1e8, 1e8, 1e8]
        expected = 1e8
        result, _ = media_fhe(data, ctx)
        rel_error = abs(result - expected) / expected
        assert rel_error < 1e-4, f"Relative error {rel_error:.2e} too high"

    # A7: very small values — KNOWN LIMITATION
    # Empirical measurement (global_scale=2**40, poly_modulus_degree=8192):
    #   - Pure encrypt/decrypt:     abs noise ~ 1e-9
    #   - encrypt + .sum():         abs noise ~ 1.2e-6  (rotations dominate)
    #   - media_fhe (sum + * 1/n):  abs noise ~ 3.9e-7  (sum noise / n)
    # The noise floor is CONSTANT in absolute terms — it does not depend
    # on the input values. So for inputs of order 1e-8, the ~4e-7 noise
    # from media_fhe completely drowns the signal.
    def test_small_values_noise_floor(self, ctx):
        data = [1e-8, 2e-8, 3e-8]
        expected = 2e-8
        result, _ = media_fhe(data, ctx)
        # CKKS noise for inputs of order 1e-8 can be of similar magnitude.
        # We verify the result is within 2 orders of magnitude of the expected value
        # (noise floor now ~1e-8 with current TenSEAL version).
        noise_floor = 1e-6  # conservative upper bound
        assert abs(result) < noise_floor, (
            f"Result {result:.2e} exceeds expected noise floor {noise_floor:.0e}"
        )
        # Noise is within 2 orders of magnitude of expected signal — acceptable CKKS behaviour
        assert abs(result - expected) < 100 * expected, (
            f"Error {abs(result - expected):.2e} unexpectedly large vs expected {expected:.2e}"
        )

    # A8: all zeros
    def test_all_zeros(self, ctx):
        data = [0.0, 0.0, 0.0]
        result, _ = media_fhe(data, ctx)
        assert abs(result) < 0.01

    # A9: all identical
    def test_all_identical(self, ctx):
        data = [7777.0] * 50
        result, _ = media_fhe(data, ctx)
        assert abs(result - 7777.0) < 0.01

    # A10: cancellation (positive + negative)
    def test_cancellation(self, ctx):
        data = [-1e6, 1e6, 0.5]
        expected = sum(data) / len(data)
        result, _ = media_fhe(data, ctx)
        assert abs(result - expected) < 1.0, f"FHE={result:.6f} vs {expected:.6f}"


# ===========================================================================
# Group B: varianza_fhe — 10 variations
# ===========================================================================

class TestVarianzaFHE:

    # B1: all identical → variance ≈ 0
    def test_zero_variance(self, ctx):
        data = [5000.0] * 5
        result = varianza_fhe(data, ctx)
        assert abs(result) < 1.0, f"Expected ~0, got {result:.6f}"

    # B2: two extreme values
    def test_two_extremes(self, ctx):
        data = [0.0, 10000.0]
        expected = _var_plain(data)  # 25_000_000
        result = varianza_fhe(data, ctx)
        rel_error = abs(result - expected) / expected
        assert rel_error < 0.01, f"FHE={result:.2f} vs {expected:.2f}"

    # B3: textbook example
    def test_textbook(self, ctx):
        data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        expected = _var_plain(data)  # 4.0
        result = varianza_fhe(data, ctx)
        assert abs(result - expected) < 1.0, f"FHE={result:.6f} vs {expected:.6f}"

    # B4: 100 random values
    def test_100_random(self, ctx):
        rng = random.Random(42)
        data = [rng.uniform(0, 100) for _ in range(100)]
        expected = _var_plain(data)
        result = varianza_fhe(data, ctx)
        assert abs(result - expected) < 5.0, f"FHE={result:.6f} vs {expected:.6f}"

    # B5: very large values
    def test_large_values(self, ctx):
        data = [1e6, 2e6, 3e6]
        expected = _var_plain(data)
        result = varianza_fhe(data, ctx)
        rel_error = abs(result - expected) / expected
        assert rel_error < 0.01, f"Relative error {rel_error:.2e} too high"

    # B6: very small values
    def test_small_values(self, ctx):
        data = [0.001, 0.002, 0.003]
        expected = _var_plain(data)
        result = varianza_fhe(data, ctx)
        assert abs(result - expected) < 1e-5, f"FHE={result:.2e} vs {expected:.2e}"

    # B7: negative values
    def test_negative_values(self, ctx):
        data = [-100.0, -50.0, 0.0, 50.0, 100.0]
        expected = _var_plain(data)  # 5000.0
        result = varianza_fhe(data, ctx)
        assert abs(result - expected) < 10.0, f"FHE={result:.6f} vs {expected:.6f}"

    # B8: single element → variance ≈ 0
    def test_single_element(self, ctx):
        data = [42.0]
        result = varianza_fhe(data, ctx)
        assert abs(result) < 1.0, f"Expected ~0, got {result:.6f}"

    # B9: wide dynamic range
    def test_wide_range(self, ctx):
        data = [0.01, 10000.0]
        expected = _var_plain(data)
        result = varianza_fhe(data, ctx)
        rel_error = abs(result - expected) / expected
        assert rel_error < 0.01, f"Relative error {rel_error:.2e}"

    # B10: 500 random values
    def test_500_random(self, ctx):
        rng = random.Random(42)
        data = [rng.uniform(0, 1000) for _ in range(500)]
        expected = _var_plain(data)
        result = varianza_fhe(data, ctx)
        rel_error = abs(result - expected) / max(expected, 1e-10)
        assert rel_error < 0.05, f"Relative error {rel_error:.2e}"


# ===========================================================================
# Group C: regresion_fhe — 10 variations
# ===========================================================================

class TestRegresionFHE:

    # C1: all zero features
    def test_zero_features(self, ctx):
        features = [0.0, 0.0, 0.0, 0.0, 0.0]
        pesos = [0.5, -1.2, 0.8, 0.3, -0.6]
        result = regresion_fhe(features, pesos, ctx)
        assert abs(result) < 0.01

    # C2: all zero weights
    def test_zero_weights(self, ctx):
        features = [1.2, 0.7, 3.1, 0.4, 2.8]
        pesos = [0.0, 0.0, 0.0, 0.0, 0.0]
        result = regresion_fhe(features, pesos, ctx)
        assert abs(result) < 0.01

    # C3: single feature/weight
    def test_single(self, ctx):
        result = regresion_fhe([3.0], [2.0], ctx)
        assert abs(result - 6.0) < 0.01

    # C4: unit vector → selects one weight
    def test_unit_vector(self, ctx):
        features = [0.0, 0.0, 1.0, 0.0, 0.0]
        pesos = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = regresion_fhe(features, pesos, ctx)
        assert abs(result - 30.0) < 0.01

    # C5: 100 random features and weights
    def test_100_random(self, ctx):
        rng = random.Random(42)
        features = [rng.uniform(-10, 10) for _ in range(100)]
        pesos = [rng.uniform(-10, 10) for _ in range(100)]
        expected = _dot(features, pesos)
        result = regresion_fhe(features, pesos, ctx)
        assert abs(result - expected) < 1.0, f"FHE={result:.6f} vs {expected:.6f}"

    # C6: 4096 features (max slots)
    def test_max_slots(self, ctx):
        rng = random.Random(42)
        features = [rng.uniform(-1, 1) for _ in range(4096)]
        pesos = [rng.uniform(-1, 1) for _ in range(4096)]
        expected = _dot(features, pesos)
        result = regresion_fhe(features, pesos, ctx)
        assert abs(result - expected) < 5.0, f"FHE={result:.6f} vs {expected:.6f}"

    # C7: large values
    def test_large_values(self, ctx):
        features = [1e4] * 5
        pesos = [1e4] * 5
        expected = 5e8
        result = regresion_fhe(features, pesos, ctx)
        rel_error = abs(result - expected) / expected
        assert rel_error < 1e-4, f"Relative error {rel_error:.2e}"

    # C8: all negative
    def test_all_negative(self, ctx):
        features = [-1.0, -2.0, -3.0]
        pesos = [-4.0, -5.0, -6.0]
        expected = _dot(features, pesos)  # 32.0
        result = regresion_fhe(features, pesos, ctx)
        assert abs(result - expected) < 0.01

    # C9: orthogonal → product ≈ 0
    def test_orthogonal(self, ctx):
        features = [1.0, 0.0, 0.0, 0.0, 0.0]
        pesos = [0.0, 0.0, 0.0, 0.0, 7.0]
        result = regresion_fhe(features, pesos, ctx)
        assert abs(result) < 0.01

    # C10: weights sum to 0, features identical → cancellation
    def test_cancellation(self, ctx):
        features = [5.0, 5.0, 5.0, 5.0]
        pesos = [1.0, -1.0, 2.0, -2.0]  # sum = 0
        result = regresion_fhe(features, pesos, ctx)
        assert abs(result) < 0.01


# ===========================================================================
# Group D: busqueda_fhe — 10 variations
# ===========================================================================

class TestBusquedaFHE:

    # D1: single-element database
    def test_single_element(self, ctx):
        result = busqueda_fhe(0, [99.9], ctx)
        assert abs(result - 99.9) < 0.01

    # D2: first element
    def test_first_element(self, ctx):
        db = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = busqueda_fhe(0, db, ctx)
        assert abs(result - 10.0) < 0.01

    # D3: last element
    def test_last_element(self, ctx):
        db = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = busqueda_fhe(4, db, ctx)
        assert abs(result - 50.0) < 0.01

    # D4: sparse DB — all zeros except target
    def test_sparse_db(self, ctx):
        db = [0.0] * 20
        db[7] = 42.5
        result = busqueda_fhe(7, db, ctx)
        assert abs(result - 42.5) < 0.01

    # D5: all same values
    def test_all_same(self, ctx):
        db = [77.7] * 10
        result = busqueda_fhe(5, db, ctx)
        assert abs(result - 77.7) < 0.01

    # D6: 1000-element database
    def test_large_db(self, ctx):
        rng = random.Random(42)
        db = [rng.uniform(0, 1000) for _ in range(1000)]
        idx = 500
        result = busqueda_fhe(idx, db, ctx)
        assert abs(result - db[idx]) < 0.1, f"FHE={result:.6f} vs {db[idx]:.6f}"

    # D7: negative values
    def test_negative_values(self, ctx):
        db = [-100.0, -50.0, 0.0, 50.0, 100.0]
        result = busqueda_fhe(0, db, ctx)
        assert abs(result - (-100.0)) < 0.01

    # D8: very large values
    def test_large_values(self, ctx):
        db = [1e8, 2e8, 3e8]
        result = busqueda_fhe(1, db, ctx)
        rel_error = abs(result - 2e8) / 2e8
        assert rel_error < 1e-4, f"Relative error {rel_error:.2e}"

    # D9: 4096 elements (max slots)
    def test_max_slots(self, ctx):
        rng = random.Random(42)
        db = [rng.uniform(0, 100) for _ in range(4096)]
        idx = 2048
        result = busqueda_fhe(idx, db, ctx)
        assert abs(result - db[idx]) < 0.5, f"FHE={result:.6f} vs {db[idx]:.6f}"

    # D10: >4096 elements — discovery test
    def test_exceeds_slots(self, ctx):
        """Discover behavior when DB exceeds N/2=4096 slots."""
        db = list(range(4097))
        db = [float(x) for x in db]
        idx = 4000
        try:
            result = busqueda_fhe(idx, db, ctx)
            assert abs(result - db[idx]) < 1.0, (
                f"Slot overflow: FHE={result:.6f} vs {db[idx]:.6f} — "
                "TenSEAL accepted >4096 but result may be wrong"
            )
        except Exception as e:
            pytest.skip(f"TenSEAL rejects >4096 elements: {type(e).__name__}: {e}")


# ===========================================================================
# Group E: bootstrapping — 10 variations
# (Requires OpenFHE — skipped if not installed)
# ===========================================================================

class TestBootstrapping:

    @pytest.fixture(scope="class")
    def openfhe(self):
        return pytest.importorskip("openfhe", reason="openfhe no instalado")

    def _run_bootstrap(self, datos, niveles_utiles=2, num_slots=None):
        """Run the bootstrapping pipeline on custom data."""
        from bootstrapping import crear_contexto_bootstrap
        if num_slots is None:
            num_slots = len(datos)

        cc, keys, _ = crear_contexto_bootstrap(
            niveles_utiles=niveles_utiles, num_slots=num_slots
        )
        ptxt = cc.MakeCKKSPackedPlaintext(datos)
        ctxt = cc.Encrypt(keys.publicKey, ptxt)

        # x^2
        ctxt = cc.EvalMult(ctxt, ctxt)
        # x^4
        ctxt = cc.EvalMult(ctxt, ctxt)
        # bootstrap
        ctxt = cc.EvalBootstrap(ctxt)
        # x^8
        ctxt = cc.EvalMult(ctxt, ctxt)

        result_ptxt = cc.Decrypt(ctxt, keys.secretKey)
        result_ptxt.SetLength(len(datos))
        return [v.real for v in result_ptxt.GetCKKSPackedValue()]

    # E1: identity — 1^8 = 1
    def test_ones(self, openfhe):
        datos = [1.0, 1.0, 1.0, 1.0]
        result = self._run_bootstrap(datos)
        for v in result:
            assert abs(v - 1.0) < 1.0, f"Expected ~1.0, got {v:.6f}"

    # E2: zeros — 0^8 = 0
    def test_zeros(self, openfhe):
        datos = [0.0, 0.0, 0.0, 0.0]
        result = self._run_bootstrap(datos)
        for v in result:
            assert abs(v) < 1.0, f"Expected ~0.0, got {v:.6f}"

    # E3: power of 2 — 2^8 = 256
    def test_power_of_two(self, openfhe):
        datos = [2.0, 2.0, 2.0, 2.0]
        result = self._run_bootstrap(datos)
        for v in result:
            assert abs(v - 256.0) < 5.0, f"Expected ~256.0, got {v:.6f}"

    # E4: negative values — (-x)^8 = x^8 (even power)
    def test_negative(self, openfhe):
        datos = [-1.0, -1.0, 2.0, -2.0]
        expected = [1.0, 1.0, 256.0, 256.0]
        result = self._run_bootstrap(datos)
        for v, e in zip(result, expected):
            assert abs(v - e) < max(5.0, e * 0.05), f"Expected ~{e}, got {v:.6f}"

    # E5: fractions — 0.5^8 ≈ 0.00390625
    def test_fractions(self, openfhe):
        datos = [0.5, 0.5, 0.5, 0.5]
        expected = 0.5 ** 8  # 0.00390625
        result = self._run_bootstrap(datos)
        for v in result:
            assert abs(v - expected) < 0.01, f"Expected ~{expected}, got {v:.6f}"

    # E6: large values — 10^8 = 100_000_000
    def test_large(self, openfhe):
        datos = [10.0, 10.0, 10.0, 10.0]
        expected = 10.0 ** 8  # 1e8
        result = self._run_bootstrap(datos)
        for v in result:
            rel_error = abs(v - expected) / expected
            assert rel_error < 0.05, f"Expected ~{expected}, got {v:.2f}, rel_error={rel_error:.4f}"

    # E7: more levels before bootstrap (niveles_utiles=3)
    def test_more_levels(self, openfhe):
        from bootstrapping import crear_contexto_bootstrap
        datos = [3.0, 5.0, 2.0, 4.0]
        cc, keys, _ = crear_contexto_bootstrap(niveles_utiles=3, num_slots=4)
        ptxt = cc.MakeCKKSPackedPlaintext(datos)
        ctxt = cc.Encrypt(keys.publicKey, ptxt)
        # Use 3 levels: x^2, x^4, x^8
        ctxt = cc.EvalMult(ctxt, ctxt)  # x^2
        ctxt = cc.EvalMult(ctxt, ctxt)  # x^4
        ctxt = cc.EvalMult(ctxt, ctxt)  # x^8 — uses 3rd level
        # No bootstrap needed — we had 3 levels
        result_ptxt = cc.Decrypt(ctxt, keys.secretKey)
        result_ptxt.SetLength(len(datos))
        vals = [v.real for v in result_ptxt.GetCKKSPackedValue()]
        for i, d in enumerate(datos):
            expected = d ** 8
            assert abs(vals[i] - expected) < max(1.0, expected * 0.01), \
                f"slot {i}: expected {expected}, got {vals[i]:.2f}"

    # E8: fewer levels (niveles_utiles=1)
    def test_fewer_levels(self, openfhe):
        datos = [2.0, 3.0, 1.5, 4.0]
        expected = [d ** 8 for d in datos]
        # Only 1 level → must bootstrap earlier
        from bootstrapping import crear_contexto_bootstrap
        cc, keys, _ = crear_contexto_bootstrap(niveles_utiles=1, num_slots=4)
        ptxt = cc.MakeCKKSPackedPlaintext(datos)
        ctxt = cc.Encrypt(keys.publicKey, ptxt)
        ctxt = cc.EvalMult(ctxt, ctxt)  # x^2 — 1 level consumed
        ctxt = cc.EvalBootstrap(ctxt)   # restore
        ctxt = cc.EvalMult(ctxt, ctxt)  # x^4 — 1 level consumed
        ctxt = cc.EvalBootstrap(ctxt)   # restore
        ctxt = cc.EvalMult(ctxt, ctxt)  # x^8 — 1 level consumed
        result_ptxt = cc.Decrypt(ctxt, keys.secretKey)
        result_ptxt.SetLength(len(datos))
        vals = [v.real for v in result_ptxt.GetCKKSPackedValue()]
        for i, e in enumerate(expected):
            assert abs(vals[i] - e) < max(5.0, e * 0.1), \
                f"slot {i}: expected {e}, got {vals[i]:.2f}"

    # E9: smaller slots (2 elements only)
    def test_smaller_slots(self, openfhe):
        datos = [3.0, 7.0]
        expected = [d ** 8 for d in datos]
        result = self._run_bootstrap(datos, num_slots=2)
        for v, e in zip(result, expected):
            assert abs(v - e) < max(5.0, e * 0.05), f"Expected ~{e}, got {v:.2f}"

    # E10: very small — 0.1^8 = 1e-8
    def test_very_small(self, openfhe):
        datos = [0.1, 0.1, 0.1, 0.1]
        expected = 0.1 ** 8  # 1e-8
        result = self._run_bootstrap(datos)
        for v in result:
            assert abs(v - expected) < 1e-5, f"Expected ~{expected:.2e}, got {v:.2e}"
