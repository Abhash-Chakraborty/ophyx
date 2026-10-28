import subprocess

from conftest import load

wd = load("kaggle/watchdog.py", "watchdog")


def fake_cli(monkeypatch, out, rc=0):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, rc, out, ""))


def test_kernel_status_parses_the_status_line(monkeypatch):
    fake_cli(monkeypatch, 'me/k has status "KernelWorkerStatus.RUNNING"\n')
    assert wd.kernel_status("k") == "RUNNING"
    fake_cli(monkeypatch, 'me/k has status "KernelWorkerStatus.COMPLETE"\n')
    assert wd.kernel_status("k") == "COMPLETE"


def test_api_errors_are_not_kernel_states(monkeypatch):
    fake_cli(monkeypatch, "429 Client Error: Too Many Requests\n", rc=1)
    assert wd.kernel_status("k") == "UNKNOWN"
    fake_cli(monkeypatch, "Cannot access kernel 'me/k' (Permission 'kernels.get' was denied).\n", rc=1)
    assert wd.kernel_status("k") == "NONE"
