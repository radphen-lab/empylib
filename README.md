# empylib

`empylib` is a Python library for computational electromagnetics, including
optical constants, Mie scattering, radiative transfer, wave optics, and
spectrometry utilities.

## Requirements

- Python 3.10 or later

## Installation

```bash
python -m pip install empylib
```

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
