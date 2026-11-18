# Backend 0.8.0 synchronization

Checked on 2026-11-18 using WSL Ubuntu and Python 3.11.

The current backend is `radphen-iadpython==0.8.0` from
`radphen-lab/iadpython` commit `239af43c380c52e7b77a1aab274407d93a838693`.
Its Python import remains `iadpython`. Empylib 0.2.0 now requires
`radphen-iadpython>=0.8.0,<0.9`; runtime guards and release metadata checks
enforce the same range and require the explicit `layers` constructor parameter.

## Source regression checks

- All eight tracked empylib test modules: 120 passed, 66 subtests passed.
- Backend CI selection, including explicit layers, mismatched indices,
  custom phase functions, and upstream/fork compatibility: 226 passed,
  28 subtests passed.
- Both editable installations in the user's WSL base environment report the
  intended versions, resolve to the local repositories, and import successfully.
- The stale upstream `iadpython.egg-info` was moved outside the editable checkout;
  no upstream distribution metadata is discoverable in that environment.

## Reproducibility

Development CI builds the exact backend commit recorded in
`tools/release/backend-source-ref.txt`. Publication continues to require a
public PyPI backend and cannot substitute the development source pin.
The earlier 0.7.0 release evidence is historical and is not approval to publish
these new artifacts. Git synchronization does not publish a PyPI release.

Existing dependency conflicts in the user's base environment were recorded
before reinstalling either editable package. Scientific source tests use a
temporary validation environment; the package installation checks use an
isolated environment.
