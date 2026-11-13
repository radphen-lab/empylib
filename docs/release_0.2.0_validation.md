# empylib 0.2.0 validation evidence

Release source: empylib `7098e3c8b3efe2051ef3a3424ae38c99e662d4ad`;
backend `5b38031c9b1d4e5763681abeb5775cadfdb543a0`.
Evidence collected on 2026-11-13. Publication is tracked separately below.

## Distribution and import

`radphen-iadpython==0.7.0` exports `iadpython` and CLI `iadp`.
empylib requires `radphen-iadpython>=0.7.0,<0.8`. Upstream `iadpython` and
this fork share installed paths and cannot coexist in one environment.
empylib rejects upstream-only and mixed installations with repair instructions.
A mixed environment must uninstall both distributions before reinstalling the fork.

## Numerical and installation evidence

- Backend full installed-wheel CI passed all ten Linux/Windows jobs on Python
  3.10–3.14: https://github.com/PanxoPanza/iadpython/actions/runs/43488521945
- Empylib candidate matrix, using the exact backend commit above:
  https://github.com/radphen-lab/empylib/actions/runs/43489095041
- Local installed full empylib suites passed on Python 3.10 with minimum
  dependencies, Python 3.10 with SciPy 1.14.1/NumPy 2, and Python 3.14 with
  current dependencies: each 115 passed, one optional reference skip,
  and 66 subtests. The affected SciPy integration paths are covered.
- Original and release backend forward calculations agree exactly across six
  HG/TABULATED/MOMENTS and boundary/diffuse cases, 816 values. Source audit
  covers 21 modules and verifies unchanged scientific data and MIT attribution.
- Backend fixes address matched-index grazing limits, Python 3.10 Gaussian
  sampling compatibility, and an inverse-fit boundary plateau. The inverse
  refinement only accepts a strictly smaller original objective value.
  Legacy API/input tests were corrected without weakening numerical assertions.
- Offline test fixtures preserve official CC0 optical data at a pinned commit;
  Mie numerical assertions are unchanged and no longer depend on external HTTP.
- Clean wheel, source archive, and independently rebuilt wheel pass metadata,
  license, Twine and all 42 bundled-resource hash checks. Builds exclude
  unrelated untracked user work and stale tracked egg-info.
- Installation checks run outside the checkout, with Git removed from PATH.
  Wheel and actual source-archive installs, pip check, upstream-only rejection,
  both conflicting installation orders, and subsequent repair are covered.

The latest clean local rebuild is in the TEMP directory
`empylib-publication-final`, with build log `empylib-publication-build.log`.
Published artifacts are rebuilt and validated by the tag workflow; their PyPI
hashes, rather than timestamp-sensitive local archive hashes, identify uploads.

## Publication

Both PyPI Trusted Publishers and protected GitHub `pypi` environments are
configured. Backend tag `v0.7.0` points to the reviewed backend commit above.
Backend publication workflow:
https://github.com/PanxoPanza/iadpython/actions/runs/43489879364

Empylib publication requires the public backend and repeats its complete
installation matrix without a candidate-source override. See
[publication instructions](../tools/release/PUBLISHING.md).
