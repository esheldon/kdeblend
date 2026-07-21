import numpy as np
import pytest
import matplotlib

from kdeblend.vis import (
    make_color_image, view_color_image, view_blend, _get_image_noise,
)

from _sims import make_blend_mbobs

matplotlib.use('Agg')

PSF_FWHMS = [1.1, 0.9, 0.8]

OBJA = dict(kind='gauss', e1=0.15, e2=-0.08, T=0.5, v=0.05, u=-1.0)
OBJB = dict(kind='gauss', e1=-0.05, e2=0.10, T=0.3, v=-0.05, u=1.0)
FLUXESA = [2.0, 3.5, 4.5, 5.0]
FLUXESB = [6.0, 5.0, 4.0, 3.5]


def _make_mbobs(nband, rng=None):
    comps_per_band = [
        [
            dict(OBJA, flux=FLUXESA[band]),
            dict(OBJB, flux=FLUXESB[band]),
        ]
        for band in range(nband)
    ]
    psf_fwhms = (PSF_FWHMS + PSF_FWHMS)[:nband]
    return make_blend_mbobs(
        comps_per_band, psf_fwhms, noise=0.01, rng=rng,
    )


def _make_objects(nband):
    """fitted-object dicts as returned by deblend, from the truth"""
    return [
        dict(type='gauss', flux=np.array(fluxes[:nband]),
             cen=np.array([obj['v'], obj['u']]),
             cen_pull=np.zeros(2),
             T=obj['T'], e1=obj['e1'], e2=obj['e2'])
        for obj, fluxes in [(OBJA, FLUXESA), (OBJB, FLUXESB)]
    ]


def test_make_color_image_three_band():
    """
    three bands map to B, G, R with the asinh stretch, clipped to
    [0, 1]
    """
    rng = np.random.RandomState(9)
    dims = (16, 16)
    imlist = [np.abs(rng.normal(size=dims)) for _ in range(3)]

    rgb = make_color_image(imlist, stretch=1.0)
    assert rgb.shape == dims + (3,)
    assert rgb.min() >= 0 and rgb.max() <= 1

    # only the reddest band set: only the R channel lights up
    zero = np.zeros(dims)
    pos = np.ones(dims) * 0.1
    rgb = make_color_image([zero, zero, pos], stretch=1.0)
    assert np.all(rgb[:, :, 0] > 0)
    assert np.all(rgb[:, :, 1] == 0)
    assert np.all(rgb[:, :, 2] == 0)

    # only the bluest: only the B channel
    rgb = make_color_image([pos, zero, zero], stretch=1.0)
    assert np.all(rgb[:, :, 0] == 0)
    assert np.all(rgb[:, :, 1] == 0)
    assert np.all(rgb[:, :, 2] > 0)


def test_make_color_image_two_band():
    """
    two bands map to B and R with G their mean
    """
    dims = (8, 8)
    zero = np.zeros(dims)
    pos = np.ones(dims) * 0.1

    rgb = make_color_image([zero, pos], stretch=1.0)
    assert np.all(rgb[:, :, 0] > 0)
    assert np.all(rgb[:, :, 2] == 0)
    # unclipped, so the G channel is exactly half the R channel
    assert np.allclose(rgb[:, :, 1], 0.5 * rgb[:, :, 0])


def test_make_color_image_one_band():
    """
    one band is grayscale; a bare 2d array is accepted
    """
    rng = np.random.RandomState(10)
    im = np.abs(rng.normal(size=(8, 8)))

    rgb = make_color_image(im, stretch=1.0)
    assert np.all(rgb[:, :, 0] == rgb[:, :, 1])
    assert np.all(rgb[:, :, 0] == rgb[:, :, 2])


def test_make_color_image_clipping():
    """
    values clip to [0, 1] for bright and negative pixels
    """
    dims = (8, 8)
    big = np.ones(dims) * 1.0e12
    neg = np.ones(dims) * (-5.0)

    rgb = make_color_image([big, big, big], stretch=1.0)
    assert np.allclose(rgb, 1.0)

    rgb = make_color_image([neg, neg, big], stretch=1.0)
    assert rgb.min() >= 0 and rgb.max() <= 1


def test_make_color_image_errors():
    """
    more than three images is an error
    """
    ims = [np.zeros((4, 4))] * 4
    with pytest.raises(ValueError):
        make_color_image(ims, stretch=1.0)


def test_get_image_noise():
    """
    the robust noise estimate recovers the sigma of a noise image
    and does not break on an empty image
    """
    rng = np.random.RandomState(11)
    sigma = 0.3
    ims = [rng.normal(scale=sigma, size=(64, 64)) for _ in range(3)]
    est = _get_image_noise(ims)
    assert np.abs(est / sigma - 1) < 0.1

    assert _get_image_noise([np.zeros((8, 8))]) > 0


def test_view_color_image(tmp_path):
    """
    view_color_image draws on a new or a sent axis and writes a file
    """
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(12)
    ims = [rng.normal(scale=0.1, size=(16, 16)) for _ in range(3)]

    fname = str(tmp_path / 'cimage.png')
    ax = view_color_image(ims, title='blah', show=False, file=fname)
    assert ax.get_title() == 'blah'
    assert (tmp_path / 'cimage.png').stat().st_size > 0

    fig, axs = plt.subplots(1, 2)
    axret = view_color_image(ims, ax=axs[1])
    assert axret is axs[1]

    plt.close('all')


@pytest.mark.parametrize('nband', [1, 2, 3])
def test_view_blend(nband, tmp_path):
    """
    view_blend makes the three-panel figure for 1-3 bands, with
    center marks and optional labels on every panel
    """
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(5)
    mbobs = _make_mbobs(nband, rng=rng)
    objects = _make_objects(nband)

    if nband == 1:
        # exercise the get_mb_obs path with a bare Observation
        obs = mbobs[0][0]
    else:
        obs = mbobs

    fname = str(tmp_path / f'blend-{nband}.png')
    fig, axs = view_blend(
        obs, objects, labels=['A', 'B'], title='the blend',
        show=False, file=fname,
    )
    assert len(axs) == 3
    for ax in axs:
        assert len(ax.lines) == len(objects)
        assert len(ax.texts) == len(objects)
    assert fig.get_suptitle() == 'the blend'
    # the default style is dark_background
    assert fig.get_facecolor()[:3] == (0, 0, 0)
    assert (tmp_path / f'blend-{nband}.png').stat().st_size > 0

    plt.close('all')


def test_view_blend_seg_boxes(tmp_path):
    """
    sending a seg map makes the 2x2 layout with the seg panel, and
    boxes are drawn as rectangles on every panel
    """
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(7)
    mbobs = _make_mbobs(3, rng=rng)
    objects = _make_objects(3)

    dims = mbobs[0][0].image.shape
    seg = np.zeros(dims, dtype='i4')
    seg[5:10, 5:10] = 1
    seg[12:16, 12:16] = 2

    boxes = [(2, 3, 10, 12), (14, 15, 8, 8)]

    fname = str(tmp_path / 'blend-seg.png')
    fig, axs = view_blend(
        mbobs, objects, seg=seg, boxes=boxes, show=False, file=fname,
    )
    assert axs.shape == (2, 2)
    assert axs[1, 1].get_title() == 'segmentation'
    assert len(axs[1, 1].images) == 1
    for ax in axs.ravel():
        assert len(ax.patches) == len(boxes)
        assert len(ax.lines) == len(objects)
    assert (tmp_path / 'blend-seg.png').stat().st_size > 0

    # boxes also work without the seg map, on the 1x3 layout, and
    # the style is selectable
    fig, axs = view_blend(
        mbobs, objects, boxes=boxes, style='default', show=False,
    )
    assert axs.shape == (3,)
    for ax in axs:
        assert len(ax.patches) == len(boxes)
    assert fig.get_facecolor()[:3] == (1, 1, 1)

    plt.close('all')


def test_view_blend_many_bands():
    """
    more than three bands requires selecting which to show
    """
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(6)
    mbobs = _make_mbobs(4, rng=rng)
    objects = _make_objects(4)

    with pytest.raises(ValueError):
        view_blend(mbobs, objects, show=False)

    with pytest.raises(ValueError):
        view_blend(mbobs, objects, bands=[0, 1, 2, 3], show=False)

    fig, axs = view_blend(mbobs, objects, bands=[0, 1, 3], show=False)
    assert len(axs) == 3

    plt.close('all')
