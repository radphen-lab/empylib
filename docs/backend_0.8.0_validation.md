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

Implementation commit: `881b17cce512ceacb43f027f34ff7ef197e45c10`.

- Clean backend wheel and sdist builds passed metadata and `twine check`;
  the sdist includes the JSON multilayer reference fixture and the wheel
  includes the `iadplus` and `iadsum` command data.
- Empylib wheel, sdist, and wheel rebuilt from the sdist passed metadata,
  resource-hash checks for all 42 bundled resources, and `twine check`.
- Outside-checkout installed regression: 119 passed, one optional reference
  test skipped, 66 subtests passed. Original wheel, rebuilt wheel, and actual
  offline sdist installs passed import/resource smoke checks and `pip check`.
- Upstream-only and both upstream/fork installation orders were rejected;
  repairing each disposable environment restored successful fork-only imports.
- Seven workflow input checks covered the default commit pin, explicit source
  and wheel overrides, publication mode, malformed refs, conflicting inputs,
  and forbidden publication overrides.

GitHub validation runs:

- [Backend Python 3.10 and 3.14 CI](https://github.com/radphen-lab/iadpython/actions/runs/44374012674)
- [Empylib distribution and cross-platform regression](https://github.com/radphen-lab/empylib/actions/runs/44374425664)

Development CI builds the exact backend commit recorded in
`tools/release/backend-source-ref.txt`. Publication continues to require a
public PyPI backend and cannot substitute the development source pin.
The earlier 0.7.0 release evidence is historical and is not approval to publish
these new artifacts. Git synchronization does not publish a PyPI release.

Existing dependency conflicts in the user's base environment were recorded
before reinstalling either editable package. Scientific source tests use a
temporary validation environment; the package installation checks use an
isolated environment.
