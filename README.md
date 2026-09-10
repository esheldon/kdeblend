# kdeblend

multi-band deblender using pre-PSF adaptive moments in
k-space with closed-form neighbor corrections.

## The idea

Each band/epoch is deconvolved by its own PSF and smoothed by a common
round gaussian, placing all the data in a common pre-seeing space (see
`ngmix.prepsfadmom`).  Objects are modeled with a per-object model
type and fit by a Jacobi sweep of adaptive-moments steps: every object
is updated from the same current state, iterated to a fixed point.
Because the moment sums are linear in the image, neighbors are never
rendered or subtracted from images: their contributions to the sums
are removed in closed form using product-gaussian identities
(~30 flops per gaussian component per step).

Structure (shape and size) is common across bands; fluxes are per band
with a common pre-seeing aperture, so colors are independent of the
per-band PSFs.

Model types:

- `'gauss'`: a single pre-PSF gaussian, fit adaptively
- `'star'`: a pre-PSF delta function ("deblended as PSF"); only the
  flux, which is linear, is fit.  Preferred for known stars: it
  removes the size-flux covariance, reducing faint-star flux scatter
  by 30-45% and removing the low-s/n flux bias.
- `'exp'`, `'dev'`: the ngmix 6/10-gaussian expansions, fit by moment
  matching.  Preferred for bright objects: unmodeled wings of a
  single-gaussian bright neighbor contaminate faint neighbors at the
  tens-of-percent level, while these models reduce this by 1-2
  orders of magnitude.
- `'bdf'`: composite exp plus dev with shared center and ellipticity
  and a per-band flux split fit by a two-aperture solve
- `'ladder'`: the free-amplitude concentric gaussian ladder
  (see `kdeblend.ladder`)

## Options

- `recenter=True`: centers update from the first moments under a
  prior pulling toward the detection positions
- `use_noise_image=True`: the per-mode noise power for the flux
  errors is measured from noise realizations attached to the
  observations; use under correlated noise, e.g. with metacal
- `full_errors=True`: the full fixed-point covariance, accounting
  for neighbor noise cross-talk and model mismatch and providing
  cross-band flux covariances for colors
- `fixed_models`: external sources subtracted in closed form
  without being fit
- `kdeblend.gpu`: CUDA batch fitter for many groups at once

## Example

```python
import numpy as np
from kdeblend import deblend

# mbobs: ngmix.MultiBandObsList with psfs set; one entry per band
# positions from detection, as sky offsets from the jacobian centers

res = deblend(
    mbobs,
    objects=[
        dict(v=0.05, u=-0.75, type='gauss', Tguess=0.4),
        dict(v=-0.05, u=0.75, type='exp', Tguess=0.4),
    ],
    fwhm_smooth=1.2,  # or None to choose from the largest psf
    rng=np.random.RandomState(8312),  # required if fwhm_smooth is None
)

for obj in res['objects']:
    print(obj['T'], obj['e1'], obj['e2'], obj['flux'])
```

## Tests

```
pytest tests
```

The GPU tests skip automatically without `cupy` and a CUDA device.

## Requirements

- `ngmix` with the `ngmix.prepsfadmom` package (currently the
  `kspace-admom` branch)
- `numpy`, `numba` (via ngmix)
- `galsim` (tests only)
- `cupy` (optional, for the GPU fitter)
- `matplotlib` (optional, for `kdeblend.vis`)


