import json

import numpy as np
from conftest import load
from safetensors.numpy import load_file, save_file

avg = load("train/average_ckpt.py", "average_ckpt")


def make_ckpt(d, value, config):
    d.mkdir()
    save_file({"w": np.full((2, 3), value, np.float32), "b": np.full(3, value, np.float16)},
              str(d / "model.safetensors"))
    (d / "config.json").write_text(json.dumps(config))
    (d / "tokenizer.json").write_text("{}")


def test_weighted_average_and_config(tmp_path):
    cfg = {"model_type": "qwen3", "rope_parameters": {"rope_theta": 100000.0, "rope_type": "default"}}
    make_ckpt(tmp_path / "a", 1.0, cfg)
    make_ckpt(tmp_path / "b", 3.0, cfg)
    out = tmp_path / "out"
    avg.average([str(tmp_path / "a"), str(tmp_path / "b")], [0.7, 0.3], str(out))

    sd = load_file(str(out / "model.safetensors"))
    assert sd["w"].dtype == np.float32 and sd["b"].dtype == np.float32
    np.testing.assert_allclose(sd["w"], 1.6, rtol=1e-6)
    np.testing.assert_allclose(sd["b"], 1.6, rtol=1e-3)

    c = json.loads((out / "config.json").read_text())
    assert c["rope_theta"] == 100000.0  # transformers < 5 reads only the top-level key
    assert c["torch_dtype"] == "float32"
    assert (out / "tokenizer.json").exists()
    assert json.loads((out / "average.json").read_text())["weights"] == [0.7, 0.3]


def test_mismatched_tensors_stop(tmp_path):
    make_ckpt(tmp_path / "a", 1.0, {})
    (tmp_path / "b").mkdir()
    save_file({"w": np.zeros((2, 3), np.float32)}, str(tmp_path / "b" / "model.safetensors"))
    try:
        avg.average([str(tmp_path / "a"), str(tmp_path / "b")], None, str(tmp_path / "out"))
    except SystemExit as e:
        assert "tensor names differ" in str(e)
    else:
        raise AssertionError("expected SystemExit")
