import numpy as np
import pytest
from conftest import load

jax = pytest.importorskip("jax")
pytest.importorskip("optax")
pt = load("train/pretrain.py", "pretrain")


def test_export_then_import_gives_the_same_params(tmp_path):
    dims = (64, 2, 4, 2, 96)  # d, layers, heads, kv heads, ffn
    params = pt.init_params(dims, 128, jax.random.PRNGKey(0))
    pt.export_hf(params, dims, 128, 256, str(tmp_path))
    back = pt.import_hf(str(tmp_path), params)
    for a, b in zip(jax.tree_util.tree_leaves(back), jax.tree_util.tree_leaves(params)):
        np.testing.assert_array_equal(a, np.asarray(b))
