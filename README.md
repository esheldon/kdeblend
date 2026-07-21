# kdeblend

multi-band deblender using pre-PSF adaptive moments in
k-space with closed-form neighbor corrections.

## Requirements

- `ngmix` with the `ngmix.prepsfadmom` package (currently the
  `kspace-admom` branch)
- `numpy`, `numba` (via ngmix)
- `galsim` (tests only)

## The idea

Each band/epoch is deconvolved by its own PSF and smoothed by a common
round gaussian, placing all the data in a common pre-seeing space (see
`ngmix.prepsfadmom`).  Objects are modeled with fixed centers and a
per-object model type, fit by a Gauss-Seidel loop of adaptive-moments
steps.  Because the moment sums are linear in the image, neighbors are
never rendered or subtracted from images: their contributions to the
sums are removed in closed form using product-gaussian identities
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
- `'exp'`: the ngmix 6-gaussian exponential expansion, fit by moment
  matching.  Preferred for bright objects: unmodeled wings of a
  single-gaussian bright neighbor contaminate faint neighbors at the
  tens-of-percent level, while the exp model reduces this by 1-2
  orders of magnitude.

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
)

for obj in res['objects']:
    print(obj['T'], obj['e1'], obj['e2'], obj['flux'])
```

## Tests

```
pytest tests
```
