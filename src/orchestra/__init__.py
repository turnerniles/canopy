"""Canopy Orchestra: a library of synthesized instruments, a score format with
odd meters, a mixer, and a world builder for adaptive game music.

    from orchestra import play, names, REGISTRY
    y = play('mallet.marimba_rosewood', 'D4', dur=0.5, vel=0.8)
"""
import importlib

from .core import (REGISTRY, register, play, names, calibrate, save_calibration,
                   nm, mtof, SR)

# Instrument families register themselves on import. Missing modules are skipped
# so the library still loads while a family is being built.
_FAMILIES = ['inst_mallets', 'inst_drums', 'inst_strings', 'inst_keys',
             'inst_winds', 'inst_voices', 'inst_creatures', 'inst_synths']
for _m in _FAMILIES:
    try:
        importlib.import_module(f'.{_m}', __name__)
    except ModuleNotFoundError as e:
        if e.name != f'{__name__}.{_m}':
            raise
