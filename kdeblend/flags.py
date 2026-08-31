"""
deblend_flags bits, in the ngmix.flags convention

NO_ATTEMPT is ngmix's, bit 0: the state of a flag word before any
deblend was attempted.  kdeblend never sets it on a result, every
returned object was attempted; it is reserved so a caller can
initialize its flag column to NO_ATTEMPT and store the kdeblend word
verbatim.  The kdeblend bits start at bit 1.
"""
import ngmix.flags
from ngmix.flags import NO_ATTEMPT

# demoted to a fixed point source after repeated failed structure
# updates; type reports 'star' and the flux is the compact
# matched-aperture flux
DEBLENDED_AS_PSF = 2**1

# structure restarted from the compact delta state after repeated
# failed structure updates
RESTARTED = 2**2

# fixed external models were subtracted from the object's sums
EXTERNALS_SUBTRACTED = 2**3

# a weight update was rejected by the stamp-size bound (a runaway,
# see deblender.MAX_WEIGHT_SIGMA_FAC)
WEIGHT_BOUNDED = 2**4

# the group hit the backstop limit on total skipped structure
# updates (100 per object) and iteration stopped early; set on
# every object of the group, whose result also has converged False
SKIP_LIMIT = 2**5

NAME_MAP = {
    NO_ATTEMPT: 'no attempt',
    DEBLENDED_AS_PSF: 'deblended as psf',
    RESTARTED: 'restarted',
    EXTERNALS_SUBTRACTED: 'externals subtracted',
    WEIGHT_BOUNDED: 'weight bounded',
    SKIP_LIMIT: 'skip limit',
}


def get_flags_str(val):
    """
    the names of the bits set in val, joined with |, as
    ngmix.flags.get_flags_str
    """
    return ngmix.flags.get_flags_str(val, name_map=NAME_MAP)
