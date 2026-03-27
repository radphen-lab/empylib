from typing import Any

__all__ = (
    'e_charge',
    'hbar',
    'speed_of_light',
    'kBoltzmann',
    'convert_units',
    'rt_style_mapper',
    'detect_spectral_spikes',
)

e_charge: float
hbar: float
speed_of_light: float
kBoltzmann: float

def convert_units(*args: Any, **kwargs: Any) -> Any: ...
def rt_style_mapper(*args: Any, **kwargs: Any) -> Any: ...
def detect_spectral_spikes(*args: Any, **kwargs: Any) -> Any: ...
