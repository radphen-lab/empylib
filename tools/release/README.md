# Release checks

The validation workflow builds from tracked files in a clean temporary source
tree (using their current contents), excluding generated egg-info and old build
outputs. It builds wheel and sdist, rebuilds a wheel from the sdist, checks
metadata and hashes every tracked package resource, and runs `twine check`.
The runtime phase installs from a downloaded wheelhouse with `--no-index` and
binary-only requirements outside the checkout. It checks imports, installed
resource contents, `pip check`, regression tests and explicit rejection of
upstream-only and mixed upstream/fork installations in both orders. Both
distributions share `iadpython` files and cannot safely coexist. Every conflict
check runs only in the temporary environment; it uninstalls both distributions
and force-reinstalls the fork to recover shared files before continuing.
The fork uses distribution name `radphen-iadpython`, import `iadpython`, and
CLI `iadp`. Each valid fork-only state receives a fresh import smoke check. PATH is restricted and checked to ensure Git is
unavailable during installation. The original wheel, rebuilt wheel and actual
source-archive install all receive installed smoke checks.
Runtime checks cap BLAS/OpenMP worker counts at one to avoid oversubscribing
memory when several validation jobs run concurrently.

Linux runs every tracked `tests/test_*.py` file on Python 3.10–3.14. Windows and
macOS run representative radiative transfer, phase moment, optical fitting and
quadrature tests on Python 3.10 and 3.14. Separate Linux jobs cover the declared
dependency floors (NumPy 1) and SciPy 1.14.1 with NumPy 2.

For prepublication integration CI, dispatch `validation.yml` with
`backend-source-ref` set to the full 40-hex commit SHA already pushed to public
`PanxoPanza/iadpython`. CI checks out that exact commit, builds its wheel, and
verifies the distribution/import metadata and absence of direct URL
requirements before the normal full matrix. This source override is exclusive
with wheel URL/SHA inputs. It is a CI candidate check, not a user installation
method. The publication workflow passes no override and always requires the
backend to be available from public PyPI.

Before the renamed backend is on PyPI, dispatch `validation.yml` with the URL of
an existing wheel release asset under
`https://github.com/PanxoPanza/iadpython/releases/download/` and its SHA256.
The helper verifies the digest, distribution name, version and package namespace.
The workflow never substitutes original `iadpython` for `radphen-iadpython`.
Without an explicit wheel, all jobs resolve `radphen-iadpython>=0.7.0,<0.8`
from public PyPI and fail if it has not been published. Merely creating the
packaging branch is insufficient; an actual wheel artifact must exist first.

For local checks (use a fresh directory for each run):

```powershell
python -m pip install build twine packaging
python tools/release/validate.py build --work C:/Temp/empylib-release-check
python tools/release/validate.py backend --work C:/Temp/empylib-release-check
python tools/release/validate.py test --work C:/Temp/empylib-release-check
```

Alternatively pass `backend --url <release-wheel-url> --sha256 <digest>`. Local
unpublished development wheels can be copied to `<work>/backend/` for local
tests, but are not accepted by the publication gate. New release tooling and
new tests must be tracked in Git before CI executes them.

GitHub environment `pypi` is configured. Register the matching PyPI Trusted
Publisher using the exact values in [PUBLISHING.md](PUBLISHING.md). After the renamed backend is public and all checks pass,
the repository owner can manually dispatch that workflow at tag `v0.2.0`,
entering `0.2.0`. It reruns the entire validation matrix against public PyPI and
publishes exactly those build artifacts. Only the final environment-protected
job receives `id-token: write`. Pushes and tag creation do not publish anything.
