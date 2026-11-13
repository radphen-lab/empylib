# Refractive-index test fixtures

These are unmodified data files from the official
[`polyanskiy/refractiveindex.info-database`](https://github.com/polyanskiy/refractiveindex.info-database)
repository, pinned to commit
[`e6d15e364f70de1cad8a96a20f75689f2fba8b21`](https://github.com/polyanskiy/refractiveindex.info-database/tree/e6d15e364f70de1cad8a96a20f75689f2fba8b21).
The repository's `LICENSE` at that commit identifies the data as CC0 1.0; its
license text is copied alongside these fixtures.

| Local fixture | Original database path | SHA-256 |
| --- | --- | --- |
| `main/TiO2/nk/Siefke.yml` | `database/data/main/TiO2/nk/Siefke.yml` | `9b03947949e54cd98f58a989bd031a5fddb1a36a9d9f7fd64d5ff2bd68c2885c` |
| `main/SiO2/nk/Franta-25C.yml` | `database/data/main/SiO2/nk/Franta-25C.yml` | `264d9c11bd19802cab5201e63a63f92b7d5e0a299ced817a3c9493302cdb1eab` |

The corresponding URLs consumed by `empylib.nklib.ri_info_data` are
`https://refractiveindex.info/database/data/main/TiO2/nk/Siefke.yml` and
`https://refractiveindex.info/database/data/main/SiO2/nk/Franta-25C.yml`.
The tests mock only those HTTP responses, then use the normal YAML parsing,
tabulated-data conversion, and interpolation path. Any other HTTP URL in this
test class raises an assertion.

`nk.CaCO3` in these tests reads the package's existing
`empylib/nk_files/CaCO3_Palik.nk` local data file; it has no HTTP fixture.
