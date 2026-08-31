"""
the deblend_flags bits follow the ngmix convention: NO_ATTEMPT is
ngmix's bit 0, the kdeblend bits are distinct and above it, and every
bit has a name
"""
import ngmix.flags

import kdeblend
from kdeblend import flags

BITS = [
    flags.DEBLENDED_AS_PSF, flags.RESTARTED, flags.EXTERNALS_SUBTRACTED,
    flags.WEIGHT_BOUNDED,
]


def test_no_attempt_is_ngmix():
    assert flags.NO_ATTEMPT is ngmix.flags.NO_ATTEMPT
    assert kdeblend.NO_ATTEMPT == 1


def test_bits_distinct_and_clear_of_no_attempt():
    assert len(set(BITS)) == len(BITS)
    for bit in BITS:
        assert bit & flags.NO_ATTEMPT == 0
        # a single bit
        assert bit & (bit - 1) == 0


def test_names():
    for bit in BITS + [flags.NO_ATTEMPT]:
        assert bit in flags.NAME_MAP
    s = flags.get_flags_str(flags.RESTARTED | flags.WEIGHT_BOUNDED)
    assert 'restarted' in s and 'weight bounded' in s
    assert kdeblend.get_flags_str(0) == flags.get_flags_str(0)


def test_deblender_reexports():
    from kdeblend import deblender
    assert deblender.DEBLENDED_AS_PSF is flags.DEBLENDED_AS_PSF
    assert deblender.WEIGHT_BOUNDED is flags.WEIGHT_BOUNDED
