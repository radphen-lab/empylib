"""Backend compatibility failures must be explicit before transport runs."""

from types import SimpleNamespace

import pytest

from empylib import rad_transfer as rt


def test_installed_backend_exposes_required_contract():
    rt._validate_iad_backend(rt._iad)
    assert rt._iad.__name__ == "iadpython"


@pytest.mark.parametrize("version", ["", "0.6.9", "0.8.0", "1.0.0", "0.7.0rc1"])
def test_backend_rejects_unsupported_versions(version):
    backend = SimpleNamespace(__version__=version, Sample=rt._iad.Sample)
    with pytest.raises(ImportError, match="pip install --upgrade"):
        rt._validate_iad_backend(backend)


@pytest.mark.parametrize("method", ["rt_matrices", "UX1_and_UXU", "unscattered_rt", "rt_diffuse_cone"])
def test_backend_rejects_missing_transport_methods(method):
    class IncompleteSample(rt._iad.Sample):
        pass

    setattr(IncompleteSample, method, None)
    with pytest.raises(ImportError, match=method):
        rt._validate_iad_backend(SimpleNamespace(__version__="0.7.0", Sample=IncompleteSample))


def test_backend_rejects_upstream_constructor():
    class UpstreamSample:
        def __init__(self, a=0, b=1, g=0, n=1):
            pass

    with pytest.raises(ImportError, match="n_sample_boundary"):
        rt._validate_iad_backend(SimpleNamespace(__version__="0.7.0", Sample=UpstreamSample))


def test_missing_backend_has_actionable_import_error():
    import subprocess
    import sys

    # A fresh interpreter exercises import-time validation rather than reusing
    # the already imported module in this pytest process.
    script = '''
import sys
sys.modules['iadpython'] = None
try:
    import empylib
except ImportError as error:
    assert 'radphen-iadpython>=0.7.0,<0.8' in str(error), str(error)
    assert 'pip install --upgrade' in str(error), str(error)
else:
    raise AssertionError('missing backend was accepted')
'''
    subprocess.run([sys.executable, "-c", script], check=True)


def _mock_distribution_versions(monkeypatch, versions):
    real_version = rt._metadata.version

    def version(name):
        if name not in {"iadpython", "radphen-iadpython"}:
            return real_version(name)
        if name not in versions:
            raise rt._metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(rt._metadata, "version", version)


def test_distribution_guard_accepts_supported_fork(monkeypatch):
    _mock_distribution_versions(monkeypatch, {"radphen-iadpython": "0.7.0"})
    rt._validate_iad_distribution()


@pytest.mark.parametrize("version", ["0.6.9", "0.8.0", "0.7.0rc1"])
def test_distribution_guard_rejects_metadata_version_independently(monkeypatch, version):
    # Valid module capabilities cannot make unsupported distribution metadata valid.
    rt._validate_iad_backend(SimpleNamespace(__version__="0.7.0", Sample=rt._iad.Sample))
    _mock_distribution_versions(monkeypatch, {"radphen-iadpython": version})
    with pytest.raises(ImportError, match="distribution version"):
        rt._validate_iad_distribution()


def test_module_version_is_checked_after_valid_distribution_metadata(monkeypatch):
    _mock_distribution_versions(monkeypatch, {"radphen-iadpython": "0.7.0"})
    rt._validate_iad_distribution()
    with pytest.raises(ImportError, match="Unsupported radphen-iadpython version"):
        rt._validate_iad_backend(SimpleNamespace(__version__="0.8.0", Sample=rt._iad.Sample))


@pytest.mark.parametrize("versions, expected", [
    ({"iadpython": "0.7.0"}, "distribution is not installed"),
    ({"iadpython": "0.7.0", "radphen-iadpython": "0.7.0"}, "Conflicting"),
    ({"radphen-iadpython": "0.8.0"}, "distribution version"),
])
def test_import_rejects_invalid_distribution_before_loading_module(versions, expected):
    import subprocess
    import sys

    # Poisoning the module also proves the metadata error occurs before import.
    script = f"""
from importlib import metadata
import sys
versions = {versions!r}
real_version = metadata.version
def version(name):
    if name not in {{"iadpython", "radphen-iadpython"}}:
        return real_version(name)
    if name not in versions:
        raise metadata.PackageNotFoundError(name)
    return versions[name]
metadata.version = version
sys.modules['iadpython'] = None
try:
    import empylib
except ImportError as error:
    message = str(error)
    assert {expected!r} in message, message
    assert 'radphen-iadpython>=0.7.0,<0.8' in message, message
    if 'Conflicting' in message:
        assert 'fresh environment' in message, message
        assert 'pip uninstall iadpython radphen-iadpython' in message, message
else:
    raise AssertionError('invalid distribution was accepted')
"""
    subprocess.run([sys.executable, "-c", script], check=True)
