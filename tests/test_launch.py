from conftest import load

launch = load("kaggle/launch.py", "launch")


def test_load_env_strips_quotes_and_comments(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# comment\nKAGGLE_API_TOKEN='abc'\nHF_TOKEN=\"x=y\"\nEMPTY=\n")
    env = launch.load_env(str(p))
    assert env == {"KAGGLE_API_TOKEN": "abc", "HF_TOKEN": "x=y", "EMPTY": ""}


def test_load_env_missing_file(tmp_path):
    assert launch.load_env(str(tmp_path / "nope")) == {}
