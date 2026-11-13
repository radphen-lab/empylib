"""Build and validate release artifacts without using the working tree at runtime.

Run with Python 3.10+; the caller supplies build, packaging and pytest.
No publication or repository mutation is performed.
"""
from __future__ import annotations

import argparse
import email
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request
import venv
import zipfile

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[2]
VERSION = "0.2.0"
BACKEND = "radphen-iadpython>=0.7.0,<0.8"
DATA_SUFFIXES = {".nk", ".txt", ".xls", ".csv"}
RELEASE_TESTS = ("tests/test_backend_requirements.py", "tests/test_scipy_integration_compat.py")


def run(*args, cwd=None, env=None):
    print("+", *map(str, args), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=True)


def tracked_files():
    return subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode().split("\0")[:-1]


def source_files():
    release_files = {path.relative_to(ROOT).as_posix()
                     for directory in ("tools/release", ".github/workflows")
                     for path in (ROOT / directory).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts}
    inputs = set(tracked_files()) | release_files | {
        name for name in RELEASE_TESTS if (ROOT / name).is_file()}
    # Generated evidence changes after the build; it is not an artifact input.
    inputs.discard("docs/release_0.2.0_validation.md")
    return sorted(inputs)


def metadata_check(raw, expected_name="empylib", expected_version=VERSION):
    metadata = email.message_from_bytes(raw)
    assert canonicalize_name(metadata["Name"]) == expected_name, metadata["Name"]
    assert metadata["Version"] == expected_version, metadata["Version"]
    python = SpecifierSet(metadata["Requires-Python"])
    assert "3.10" in python and "3.14" in python
    assert "3.9" not in python and "3.15" not in python
    requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
    assert all(req.url is None for req in requirements), "Direct URL dependency in metadata"
    assert all(canonicalize_name(req.name) != "iadpython" for req in requirements)
    backend = [req for req in requirements if canonicalize_name(req.name) == "radphen-iadpython"]
    assert len(backend) == 1
    assert backend[0].specifier == SpecifierSet(">=0.7.0,<0.8")
    assert not backend[0].marker, "Backend must not be optional or platform-conditional"
    return metadata


def data_manifest():
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in tracked_files()
        if name == "empylib/py.typed"
        or (name.startswith("empylib/") and Path(name).suffix in DATA_SUFFIXES)
    }


def validate_archive(path, manifest):
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            files = {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}
        metadata_check(next(value for name, value in files.items() if name.endswith(".dist-info/METADATA")))
        assert any(name.endswith("/LICENCE.txt") for name in files), "Wheel license missing"
    else:
        with tarfile.open(path) as archive:
            files = {
                member.name.split("/", 1)[1]: archive.extractfile(member).read()
                for member in archive.getmembers() if member.isfile() and "/" in member.name
            }
        metadata_check(files["PKG-INFO"])
        assert all(name in files for name in ("LICENCE.txt", "README.md", "CHANGELOG.txt"))
    assert manifest, "No tracked package data found"
    for name, digest in manifest.items():
        assert name in files, f"Missing package data: {name} in {path}"
        assert hashlib.sha256(files[name]).hexdigest() == digest, f"Changed package data: {name}"
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in files)
    print(f"Validated metadata and {len(manifest)} resources: {path.name}")


def build(work):
    source = work / "source"
    source.mkdir(parents=True)
    source_manifest = {}
    for name in source_files():
        if ".egg-info/" in name or name.startswith(("build/", "dist/")):
            continue
        origin = ROOT / name
        if origin.is_file():
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)
            source_manifest[name] = hashlib.sha256(origin.read_bytes()).hexdigest()
    (work / "source-manifest.json").write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")
    dist = work / "dist"
    run(sys.executable, "-m", "build", source, "--outdir", dist)
    manifest = data_manifest()
    sdist, = dist.glob("*.tar.gz")
    wheel, = dist.glob("*.whl")
    validate_archive(sdist, manifest)
    validate_archive(wheel, manifest)
    unpack = work / "unpacked"
    unpack.mkdir()
    with tarfile.open(sdist) as archive:
        for member in archive.getmembers():
            destination = (unpack / member.name).resolve()
            assert destination.is_relative_to(unpack.resolve()), member.name
            assert not member.issym() and not member.islnk(), member.name
        archive.extractall(unpack)
    extracted, = unpack.iterdir()
    rebuilt = work / "rebuilt"
    run(sys.executable, "-m", "build", extracted, "--wheel", "--outdir", rebuilt)
    rebuilt_wheel, = rebuilt.glob("*.whl")
    validate_archive(rebuilt_wheel, manifest)
    run(sys.executable, "-m", "twine", "check", *dist.iterdir(), rebuilt_wheel)
    (work / "data-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def fetch_backend(work, url, digest):
    destination = work / "backend"
    destination.mkdir(parents=True, exist_ok=True)
    if not url:
        assert not digest, "SHA256 requires a backend URL"
        run(sys.executable, "-m", "pip", "--isolated", "download", "--index-url", "https://pypi.org/simple",
            "--only-binary=:all:", "--no-deps", "--dest", destination, BACKEND)
        wheel, = destination.glob("*.whl")
        validate_backend_wheel(wheel)
        return
    parsed = urllib.parse.urlparse(url)
    assert parsed.scheme == "https" and parsed.netloc == "github.com"
    assert parsed.path.startswith("/PanxoPanza/iadpython/releases/download/") and parsed.path.endswith(".whl")
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest.lower())
    wheel = destination / Path(parsed.path).name
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    assert hashlib.sha256(payload).hexdigest() == digest.lower(), "Backend SHA256 mismatch"
    wheel.write_bytes(payload)
    validate_backend_wheel(wheel)
    print(f"Verified prerelease backend wheel: {wheel.name}")


def validate_backend_wheel(wheel):
    """Check either a downloaded wheel or a wheel built from a pinned CI source."""
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata = email.message_from_bytes(archive.read(next(n for n in names if n.endswith(".dist-info/METADATA"))))
        assert canonicalize_name(metadata["Name"]) == "radphen-iadpython"
        assert metadata["Version"] in SpecifierSet(">=0.7.0,<0.8")
        requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
        assert all(req.url is None for req in requirements), "Direct URL backend dependency"
        assert any(n.startswith("iadpython/") for n in names)
        assert not any(n.startswith("radphen_iadpython/") for n in names), "Unexpected renamed import package"
    print(f"Validated backend distribution metadata and iadpython import package: {wheel.name}")


def python_in(environment):
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def runtime_env(work):
    python = python_in(work / "installed")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(PIP_NO_INDEX="1", PIP_FIND_LINKS=str(work / "wheelhouse"), MPLBACKEND="Agg",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    paths = [str(python.parent)]
    if os.name == "nt":
        paths.append(str(Path(os.environ["SystemRoot"]) / "System32"))
    env["PATH"] = os.pathsep.join(paths)
    assert shutil.which("git", path=env["PATH"]) is None, "Git unexpectedly present in isolated PATH"
    return env


def installed_smoke(manifest_file):
    import importlib
    import importlib.resources
    import numpy as np
    import empylib
    import iadpython

    assert empylib.__version__ == VERSION
    assert Path(empylib.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), empylib.__file__
    assert iadpython.__name__ == "iadpython"
    assert empylib.rad_transfer._iad is iadpython
    assert importlib.util.find_spec("radphen_iadpython") is None
    assert importlib.metadata.version("radphen-iadpython") in SpecifierSet(">=0.7.0,<0.8")
    for name in ("color_system", "dense_spheres", "miescattering", "nklib", "rad_transfer",
                 "ref_spectra", "scuffem", "spectrometry", "utils", "waveoptics"):
        importlib.import_module(f"empylib.{name}")
    manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
    for name, digest in manifest.items():
        resource = importlib.resources.files("empylib").joinpath(name.removeprefix("empylib/"))
        assert hashlib.sha256(resource.read_bytes()).hexdigest() == digest, name
    from empylib.ref_spectra import AM15, T_atmosphere, yCIE_lum
    for function in (AM15, T_atmosphere, yCIE_lum):
        assert np.isfinite(function(np.array([0.5, 0.6]))).all(), function.__name__
    print("Installed module imports and packaged resource smoke passed")


def test(work, selection, constraint):
    # Resolve everything before installing offline. No Git or GitHub is needed
    # for the actual installation, including all transitive dependencies.
    wheelhouse = work / "wheelhouse"
    wheelhouse.mkdir()
    backend, = (work / "backend").glob("*.whl")
    wheel, = (work / "dist").glob("*.whl")
    args = ["--constraint", constraint] if constraint else []
    run(sys.executable, "-m", "pip", "--isolated", "download", "--index-url", "https://pypi.org/simple",
        "--only-binary=:all:", "--dest", wheelhouse, *args, backend, wheel, "pytest", "setuptools>=77", "wheel")
    environment = work / "installed"
    venv.create(environment, with_pip=True)
    python = python_in(environment)
    outside = work / "outside-checkout"
    outside.mkdir()
    # All subprocess Python paths are absolute. Restrict PATH to the venv
    # (and Windows system DLL tools), proving Git is unavailable to pip.
    env = runtime_env(work)
    run(python, "-m", "pip", "install", "--no-compile", "--no-index", "--only-binary=:all:",
        "--find-links", wheelhouse, wheel, "pytest", cwd=outside, env=env)
    run(python, "-m", "pip", "check", cwd=outside, env=env)
    run(python, "-m", "pip", "list", "--format=json", cwd=outside, env=env)
    regression(work, selection)


def regression(work, selection):
    """Run or resume installed checks using an already prepared wheelhouse/venv."""
    python = python_in(work / "installed")
    outside = work / "outside-checkout"
    env = runtime_env(work)
    wheel, = (work / "dist").glob("*.whl")
    run(python, "-m", "pip", "install", "--no-deps", "--no-compile", "--no-index", "--force-reinstall",
        wheel, cwd=outside, env=env)
    # Copy this helper so sys.path[0] cannot point back into the checkout.
    helper = outside / "validate_installed.py"
    shutil.copy2(__file__, helper)
    run(python, helper, "smoke", "--manifest", work / "data-manifest.json", cwd=outside, env=env)
    test_dir = outside / "tests"
    test_dir.mkdir(exist_ok=True)
    tracked = [name for name in source_files() if name.startswith("tests/test_") and name.endswith(".py")]
    if selection == "representative":
        tracked = [name for name in tracked if Path(name).name in {
            "test_rad_transfer_sanity.py", "test_phase_function_moments.py",
            "test_nklib_fit_to_oscillator.py", "test_riccati_and_quadrature.py",
            "test_backend_requirements.py", "test_scipy_integration_compat.py"}]
    assert tracked, "No tracked regression tests selected"
    for name in tracked:
        shutil.copy2(ROOT / name, test_dir / Path(name).name)
    run(python, "-m", "pytest", "-q", "--import-mode=importlib", test_dir, cwd=outside, env=env)
    finish(work, python, outside, env)


def finish(work, python=None, outside=None, env=None):
    """Finish artifact checks; separately callable after a completed test run."""
    wheelhouse = work / "wheelhouse"
    backend, = (work / "backend").glob("*.whl")
    python = python or python_in(work / "installed")
    outside = outside or work / "outside-checkout"
    helper = outside / "validate_installed.py"
    if env is None:
        env = runtime_env(work)
    # The distributions share iadpython files. Exercise conflicts only here in
    # a disposable environment; after every conflict uninstall BOTH before
    # force-reinstalling the fork, since either uninstall removes shared files.
    upstream_candidates = list(wheelhouse.glob("iadpython-*.whl"))
    if not upstream_candidates:
        run(sys.executable, "-m", "pip", "--isolated", "download", "--index-url", "https://pypi.org/simple",
            "--only-binary=:all:", "--no-deps", "--dest", wheelhouse, "iadpython==0.5.3")
    upstream, = wheelhouse.glob("iadpython-*.whl")
    reject = """try:
    import empylib.rad_transfer
except ImportError as error:
    message = str(error)
    assert 'radphen-iadpython' in message, message
    print('Expected backend rejection:', message)
else:
    raise AssertionError('Invalid solver installation was accepted')
"""
    def clean_solvers():
        run(python, "-m", "pip", "uninstall", "-y", "iadpython", "radphen-iadpython", cwd=outside, env=env)

    def recover_fork():
        clean_solvers()
        run(python, "-m", "pip", "install", "--no-deps", "--no-compile", "--no-index", "--force-reinstall",
            backend, cwd=outside, env=env)
        run(python, helper, "smoke", "--manifest", work / "data-manifest.json", cwd=outside, env=env)

    clean_solvers()
    run(python, "-m", "pip", "install", "--no-deps", "--no-compile", "--no-index", upstream, cwd=outside, env=env)
    run(python, "-c", reject, cwd=outside, env=env)
    recover_fork()
    for index, order in enumerate(((upstream, backend), (backend, upstream))):
        print(f"Conflict rejection order {index}: {[package.name for package in order]}", flush=True)
        clean_solvers()
        try:
            for package in order:
                run(python, "-m", "pip", "install", "--no-deps", "--no-compile", "--no-index",
                    package, cwd=outside, env=env)
            run(python, "-c", reject.replace("assert 'radphen-iadpython' in message, message",
                "assert 'Conflicting iadpython and radphen-iadpython' in message, message"), cwd=outside, env=env)
        finally:
            recover_fork()
    run(python, "-m", "pip", "check", cwd=outside, env=env)
    # Validate installation of the independently rebuilt wheel too.
    rebuilt, = (work / "rebuilt").glob("*.whl")
    run(python, "-m", "pip", "install", "--no-compile", "--no-index", "--no-deps", "--force-reinstall", rebuilt,
        cwd=outside, env=env)
    run(python, helper, "smoke", "--manifest", work / "data-manifest.json", cwd=outside, env=env)
    # Force an actual sdist installation (with isolated offline build deps),
    # independently of both prebuilt wheels and without access to the checkout.
    sdist, = (work / "dist").glob("*.tar.gz")
    run(python, "-m", "pip", "uninstall", "-y", "empylib", cwd=outside, env=env)
    run(python, "-m", "pip", "install", "--no-compile", "--no-index", "--no-cache-dir", "--find-links", wheelhouse,
        sdist, cwd=outside, env=env)
    run(python, "-m", "pip", "check", cwd=outside, env=env)
    run(python, helper, "smoke", "--manifest", work / "data-manifest.json", cwd=outside, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "backend", "test", "regression", "finish", "smoke", "backend-check"))
    parser.add_argument("--work", type=Path)
    parser.add_argument("--url", default="")
    parser.add_argument("--sha256", default="")
    parser.add_argument("--selection", choices=("full", "representative"), default="full")
    parser.add_argument("--constraint")
    parser.add_argument("--manifest")
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    if args.command == "smoke":
        installed_smoke(args.manifest)
        return
    if args.command == "backend-check":
        assert args.wheel is not None, "--wheel is required"
        validate_backend_wheel(args.wheel)
        return
    assert args.work is not None, "--work is required"
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if args.command == "build":
        build(work)
    elif args.command == "backend":
        fetch_backend(work, args.url, args.sha256)
    elif args.command == "finish":
        finish(work)
    elif args.command == "regression":
        regression(work, args.selection)
    else:
        test(work, args.selection, args.constraint)


if __name__ == "__main__":
    main()
