# empylib

`empylib` is a Python library for computational electromagnetics, including
optical constants, Mie scattering, radiative transfer, wave optics, and
spectrometry utilities.

## Requirements

- Python 3.10–3.14

## Installation

```bash
python -m pip install empylib==0.2.0
```

This command will be available after empylib 0.2.0 is published to PyPI.
Before both packages are published, install the fork wheel first, then install
empylib from a local checkout. Build the fork wheel from its source checkout
and replace the placeholder below with the wheel's actual path:

```bash
python -m pip install /path/to/radphen_iadpython-0.7.0-py3-none-any.whl
cd /path/to/empylib
python -m pip install .
```

empylib 0.2.0 depends on the `radphen-iadpython>=0.7.0,<0.8` fork, which
provides the `iadpython` import package and the `iadp` command. The distribution
name is `radphen-iadpython`; Python code continues to use `import iadpython`. The fork must be published before empylib
0.2.0 so pip can resolve the dependency.

If you currently install empylib directly from Git, migrate to the published
release with:

```bash
python -m pip uninstall empylib
python -m pip install empylib==0.2.0
```

Use a fresh virtual environment. Upstream `iadpython` and the fork
`radphen-iadpython` are mutually exclusive in one environment: both install
the same `iadpython` package files and `iadp` command. empylib rejects upstream
alone and rejects environments containing both distributions. To repair a
mixed environment, uninstall both distributions, then reinstall the fork
before installing empylib.
For release validation and build details, see the [release guide](tools/release/README.md).

## Example

Load tabulated optical constants bundled with the package:

```python
import numpy as np
from empylib.nklib import get_nkfile

wavelength_um = np.linspace(0.40, 2.00, 250)
refractive_index, source_data = get_nkfile(
    wavelength_um,
    "sio2_Palik_Lemarchand2013",
    get_from_local_path=True,
)
```

See the [project repository](https://github.com/radphen-lab/empylib) for
tutorial notebooks and further examples.

## License

empylib is released under the [MIT License](LICENCE.txt).
