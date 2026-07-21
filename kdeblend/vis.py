"""
visualization of blends and deblending results

Color images are asinh composites (Lupton et al. 2004) built from up
to three band images ordered bluest to reddest: one band is shown in
grayscale, two as a blue/red composite with the green channel set to
their mean, and three as RGB.

This module needs matplotlib, and galsim via kdeblend.render; these
are not dependencies of the package and it is not imported by the
package __init__.  Use

    from kdeblend import vis
"""
import numpy as np

from .render import render_model

DEFAULT_Q = 8.0
DEFAULT_STRETCH_FAC = 4.0


def view_color_image(
    imlist, stretch=None, Q=DEFAULT_Q,
    ax=None, title=None, extent=None, show=True, file=None,
):
    """
    Display a color image built from up to three band images.

    Parameters
    ----------
    imlist: array or list of arrays
        One to three band images, ordered bluest to reddest.
    stretch: float, optional
        The asinh stretch; if not sent, DEFAULT_STRETCH_FAC times a
        robust estimate of the noise from the images themselves.
    Q: float, optional
        The asinh softening, default 8.
    ax: matplotlib axis, optional
        Draw onto this axis; a new figure is created if not sent.
    title: str, optional
        Title for the axis.
    extent: tuple, optional
        Passed to imshow.
    show: bool, optional
        Show the figure, default True; ignored when ax is sent.
    file: str, optional
        Write the figure to this file; ignored when ax is sent.

    Returns
    -------
    the matplotlib axis
    """
    import matplotlib.pyplot as plt

    imlist = _as_imlist(imlist)
    if stretch is None:
        stretch = DEFAULT_STRETCH_FAC * _get_image_noise(imlist)

    rgb = make_color_image(imlist, stretch=stretch, Q=Q)

    created = ax is None
    if created:
        fig, ax = plt.subplots(layout='constrained')

    ax.imshow(rgb, origin='lower', extent=extent)
    if title is not None:
        ax.set_title(title)

    if created:
        if file is not None:
            fig.savefig(file, dpi=150)
        if show:
            plt.show()

    return ax


def view_blend(
    obs, objects, bands=None, stretch=None, Q=DEFAULT_Q,
    labels=None, seg=None, boxes=None, title=None,
    style='dark_background', show=True, file=None,
):
    """
    Show color images of the data, the real-space model and the
    data - model residual for a deblending result, all with the same
    asinh mapping, optionally with a segmentation map panel and
    boxes drawn on every panel.  The object centers are marked on
    every panel.

    The fitted models are rendered pre-psf and convolved with each
    shown band's psf.  Only the first epoch of each band is shown.

    Parameters
    ----------
    obs: Observation, ObsList, or MultiBandObsList
        The observation(s) that were deblended.
    objects: list of dicts
        The fitted objects, the 'objects' entry of the deblend
        result.
    bands: sequence of int, optional
        Which bands to show, at most three, ordered bluest to
        reddest.  Defaults to all bands; must be sent if there are
        more than three.
    stretch: float, optional
        The asinh stretch; if not sent, DEFAULT_STRETCH_FAC times the
        median noise from the weight maps.
    Q: float, optional
        The asinh softening, default 8.
    labels: list of str, optional
        Per-object labels to annotate at the centers on every
        panel.
    seg: array, optional
        A segmentation map with the shape of the shown images; when
        sent the layout becomes 2x2, with the seg map shown in
        categorical colors on a black background as the fourth
        panel.
    boxes: sequence, optional
        Regions (row_start, col_start, nrow, ncol) in pixel
        coordinates of the shown images, drawn as rectangles on
        every panel.
    title: str, optional
        A figure suptitle.  Prefer this over calling fig.suptitle
        afterwards, which would not be covered by the style context.
    style: str, optional
        A matplotlib style name, used with plt.style.context around
        the figure creation; default 'dark_background'.  Note text
        artists added to the figure afterwards are not covered by
        the context.
    show: bool, optional
        Show the figure, default True.
    file: str, optional
        Write the figure to this file.

    Returns
    -------
    fig, axs
        axs has shape (3,) without a seg map, (2, 2) with one
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from ngmix.observation import get_mb_obs

    mbobs = get_mb_obs(obs)
    nband = len(mbobs)

    if bands is None:
        if nband > 3:
            raise ValueError(
                f'got {nband} bands; send bands= to choose at most '
                'three to show'
            )
        bands = list(range(nband))
    if len(bands) > 3:
        raise ValueError(f'send at most three bands, got {len(bands)}')

    obslist = [mbobs[band][0] for band in bands]
    data_ims = [tobs.image for tobs in obslist]
    model_ims = [
        render_model(objects, tobs, band)
        for tobs, band in zip(obslist, bands)
    ]
    resid_ims = [d - m for d, m in zip(data_ims, model_ims)]

    if stretch is None:
        stretch = DEFAULT_STRETCH_FAC * _get_obs_noise(obslist)

    extent = _get_extent(obslist[0])

    with plt.style.context(style):
        if seg is None:
            fig, axs = plt.subplots(
                1, 3, figsize=(12, 4.4), layout='constrained',
            )
        else:
            fig, axs = plt.subplots(
                2, 2, figsize=(9.5, 9), layout='constrained',
            )

        if title is not None:
            fig.suptitle(title)

        panels = [
            ('data', data_ims), ('model', model_ims),
            ('data - model', resid_ims),
        ]
        for ax, (ptitle, ims) in zip(axs.ravel(), panels):
            view_color_image(
                ims, stretch=stretch, Q=Q, ax=ax, title=ptitle,
                extent=extent,
            )

        if seg is not None:
            sax = axs.ravel()[3]
            sax.imshow(
                seg_color_image(seg), origin='lower', extent=extent,
                interpolation='nearest',
            )
            sax.set_title('segmentation')

        for ax in axs.ravel():
            ax.set_xlabel('u')
            ax.set_ylabel('v')

        for i, obj in enumerate(objects):
            v, u = obj['cen']
            for ax in axs.ravel():
                ax.plot(u, v, '+', color='white', ms=8, mew=0.8)
                if labels is not None:
                    ax.annotate(
                        labels[i], (u, v),
                        xytext=(3, 3), textcoords='offset points',
                        color='white', fontsize=9,
                    )

        if boxes is not None:
            jrow, jcol = obslist[0].jacobian.get_cen()
            scale = obslist[0].jacobian.get_scale()
            for ax in axs.ravel():
                for row_start, col_start, nrow, ncol in boxes:
                    ax.add_patch(Rectangle(
                        ((col_start - 0.5 - jcol) * scale,
                         (row_start - 0.5 - jrow) * scale),
                        ncol * scale, nrow * scale,
                        fill=False, edgecolor='yellow', linewidth=0.8,
                        alpha=0.7,
                    ))

        if file is not None:
            fig.savefig(file, dpi=150)
        if show:
            plt.show()

    return fig, axs


def make_color_image(imlist, stretch, Q=DEFAULT_Q):
    """
    Make an asinh color composite from up to three band images
    ordered bluest to reddest.

    Parameters
    ----------
    imlist: array or list of arrays
        One to three band images.  One band maps to grayscale, two to
        the B and R channels with G their mean, three to B, G, R.
    stretch: float
        The asinh stretch.
    Q: float, optional
        The asinh softening, default 8.

    Returns
    -------
    rgb: (nrow, ncol, 3) array with values clipped to [0, 1]
    """
    imlist = _as_imlist(imlist)
    nband = len(imlist)

    if nband == 1:
        imb = img = imr = imlist[0]
    elif nband == 2:
        imb, imr = imlist
        img = 0.5 * (imb + imr)
    elif nband == 3:
        imb, img, imr = imlist
    else:
        raise ValueError(f'send at most three images, got {nband}')

    inten = (imr + img + imb) / 3
    with np.errstate(invalid='ignore', divide='ignore'):
        fac = np.arcsinh(Q * inten / stretch) / (Q * inten)
    fac = np.where(inten > 0, fac, 0.0)

    return np.stack(
        [np.clip(im * fac, 0, 1) for im in (imr, img, imb)], axis=-1,
    )


def seg_color_image(seg):
    """
    Make a categorical color image of a segmentation map, with the
    background black.

    Parameters
    ----------
    seg: array
        A segmentation map: integers, with 0 the background.

    Returns
    -------
    rgb: (nrow, ncol, 3) array
    """
    import matplotlib.pyplot as plt

    nseg = seg.max()
    colors = plt.cm.tab20(np.arange(nseg) % 20)[:, :3]
    rgb = np.zeros(seg.shape + (3,))
    for i in range(1, nseg + 1):
        rgb[seg == i] = colors[i - 1]
    return rgb


def _get_extent(obs):
    """imshow extent in sky coordinates relative to the jacobian
    center, assuming a nearly diagonal jacobian"""
    nrow, ncol = obs.image.shape
    jrow, jcol = obs.jacobian.get_cen()
    scale = obs.jacobian.get_scale()
    return (
        (-0.5 - jcol) * scale, (ncol - 0.5 - jcol) * scale,
        (-0.5 - jrow) * scale, (nrow - 0.5 - jrow) * scale,
    )


def _as_imlist(imlist):
    """wrap a single image in a list"""
    if isinstance(imlist, np.ndarray) and imlist.ndim == 2:
        return [imlist]
    return list(imlist)


def _get_image_noise(imlist):
    """robust noise sigma estimate from the images themselves"""
    sigmas = [
        1.4826 * np.median(np.abs(im - np.median(im)))
        for im in imlist
    ]
    sigma = np.median(sigmas)
    if not np.isfinite(sigma) or sigma <= 0:
        imax = max(np.abs(im).max() for im in imlist)
        sigma = imax / 100 if imax > 0 else 1.0
    return sigma


def _get_obs_noise(obslist):
    """median noise sigma from the weight maps, falling back to the
    images when no weights are set"""
    sigmas = []
    for obs in obslist:
        w = obs.weight
        if np.any(w > 0):
            sigmas.append(np.median(1 / np.sqrt(w[w > 0])))
    if len(sigmas) == 0:
        return _get_image_noise([obs.image for obs in obslist])
    return np.median(sigmas)
