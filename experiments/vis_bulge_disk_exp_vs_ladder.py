"""a bulge+disk galaxy (red bulge, blue disk) fit with exp and ladder:
color images, residuals and profiles against the noiseless truth"""
import sys, numpy as np
sys.path.insert(0, '/home/esheldon/git/kdeblend/tests')
from _sims import make_blend_mbobs, SCALE
from kdeblend import deblend
from kdeblend.render import render_model
from kdeblend.vis import make_color_image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = sys.argv[1]
rng = np.random.RandomState(5)
dim = 128; psf_fwhms = [0.9, 0.85, 0.8]; noise = 0.9
bands = ['g', 'r', 'i']
disk = dict(kind='exp', hlr=1.2, e1=0.2, e2=0.1, v=0.0, u=0.0)
bulge = dict(kind='sersic', n=4.0, hlr=0.5, e1=0.05, e2=0.02, v=0.0, u=0.0)
disk_flux = [600.0, 800.0, 1000.0]
bulge_flux = [150.0, 350.0, 600.0]
comps = [[dict(disk, flux=disk_flux[b]), dict(bulge, flux=bulge_flux[b])] for b in range(3)]
mbobs = make_blend_mbobs(comps, psf_fwhms, noise=noise, rng=rng, dim=dim)
truth = make_blend_mbobs(comps, psf_fwhms, dim=dim)
data = [mbobs[b][0].image for b in range(3)]
true_im = [truth[b][0].image for b in range(3)]

fits = {}
for t in ('exp', 'ladder'):
    res = deblend(mbobs, [dict(v=0.0, u=0.0, type=t, Tguess=0.8)], rng=np.random.RandomState(0))
    o = res['objects'][0]
    model = [render_model(res['objects'], mbobs[b][0], b, Tsmooth=res['Tsmooth']) for b in range(3)]
    fits[t] = dict(res=res, obj=o, model=model)
    print(f"{t}: converged {res['converged']} in {res['numiter']} sweeps; T {o['T']:.3f} e1 {o['e1']:.3f} e2 {o['e2']:.3f}; flux {np.round(o['flux'], 1)} s2n {o['s2n']:.1f}"
          + (f"; total_flux {np.round(o['total_flux'], 1)} fixed_flux {np.round(o['fixed_flux'], 1)} gradient {np.round(o['gradient'], 3)}" if t == 'ladder' else ''))
print('true total flux per band:', [disk_flux[b] + bulge_flux[b] for b in range(3)])

# radial profiles in circular annuli about the center (i band, and g-i color)
cen = (dim - 1) / 2
yy, xx = np.mgrid[:dim, :dim]
r = np.hypot(yy - cen, xx - cen) * SCALE
edges = np.arange(0, 6.0 + 1e-9, 0.25)
rmid = 0.5 * (edges[1:] + edges[:-1])
def prof(im):
    return np.array([im[(r >= lo) & (r < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])])
def color(ims):
    g, i = prof(ims[0]), prof(ims[2])
    with np.errstate(invalid='ignore', divide='ignore'):
        return -2.5 * np.log10(g / i)
p_true = prof(true_im[2]); p_data = prof(data[2])
p_fit = {t: prof(fits[t]['model'][2]) for t in fits}
c_true = color(true_im); c_data = color(data)
c_fit = {t: color(fits[t]['model']) for t in fits}
# reduced chi2 of the residual within r < 3 arcsec, i band
mask = r < 3.0
for t in fits:
    chi2 = (((data[2] - fits[t]['model'][2]) ** 2)[mask] / noise ** 2).sum() / mask.sum()
    fits[t]['chi2'] = chi2
    print(f"{t}: reduced chi2 (i band, r < 3 arcsec) {chi2:.3f}; model-truth flux within 3 arcsec {(fits[t]['model'][2][mask].sum() / true_im[2][mask].sum() - 1) * 100:+.2f} percent")

# figure
stretch = 4.0 * noise
fig = plt.figure(figsize=(15, 8.6), facecolor='white')
gs0 = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.95], hspace=0.22, left=0.05, right=0.99, top=0.92, bottom=0.08)
gs_top = gs0[0].subgridspec(1, 5, wspace=0.08)
gs_bot = gs0[1].subgridspec(1, 3, width_ratios=[2.0, 2.0, 1.4], wspace=0.3)
panels = [('data', data), ('exp model', fits['exp']['model']), ('data - exp', [d - m for d, m in zip(data, fits['exp']['model'])]),
          ('ladder model', fits['ladder']['model']), ('data - ladder', [d - m for d, m in zip(data, fits['ladder']['model'])])]
for k, (title, ims) in enumerate(panels):
    ax = fig.add_subplot(gs_top[0, k])
    rgb = make_color_image(ims, stretch=stretch)
    ax.imshow(rgb, origin='lower', extent=[-cen * SCALE, cen * SCALE] * 2)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
    ax.set_xticks([]); ax.set_yticks([])
ax = fig.add_subplot(gs_bot[0, 0])
ax.plot(rmid, p_true, 'k-', lw=2, label='truth (noiseless)')
ax.plot(rmid, p_data, 'o', color='0.4', ms=4, label='data')
ax.plot(rmid, p_fit['exp'], '--', color='tab:red', lw=2, label=f"exp  ($\\chi^2_\\nu$={fits['exp']['chi2']:.2f})")
ax.plot(rmid, p_fit['ladder'], '-', color='tab:blue', lw=2, label=f"ladder  ($\\chi^2_\\nu$={fits['ladder']['chi2']:.2f})")
ax.set_yscale('log'); ax.set_ylim(noise * 0.3, p_true.max() * 1.5)
ax.axhline(noise, color='0.6', ls=':', lw=1); ax.text(5.9, noise * 1.15, 'pixel noise', color='0.5', ha='right', fontsize=8)
ax.set_xlabel('r [arcsec]'); ax.set_ylabel('i-band surface brightness [per pixel]')
ax.set_title('azimuthal profile (i band)', fontsize=11); ax.legend(fontsize=9, loc='upper right')
ax = fig.add_subplot(gs_bot[0, 1])
ok = p_true > 0.5 * noise
for t, c, ls in (('exp', 'tab:red', '--'), ('ladder', 'tab:blue', '-')):
    ax.plot(rmid[ok], (p_fit[t][ok] / p_true[ok] - 1) * 100, ls, color=c, lw=2, label=t)
ax.plot(rmid[ok], (p_data[ok] / p_true[ok] - 1) * 100, 'o', color='0.4', ms=3, label='data')
ax.axhline(0, color='k', lw=1); ax.set_ylim(-80, 80)
ax.set_xlabel('r [arcsec]'); ax.set_ylabel('(model - truth) / truth  [\\%]')
ax.set_title('profile residual vs the truth (i band)', fontsize=11); ax.legend(fontsize=9)
ax = fig.add_subplot(gs_bot[0, 2])
okc = p_true > 2 * noise
ax.plot(rmid[okc], c_true[okc], 'k-', lw=2, label='truth')
ax.plot(rmid[okc], c_data[okc], 'o', color='0.4', ms=3, label='data')
ax.plot(rmid[okc], c_fit['exp'][okc], '--', color='tab:red', lw=2, label='exp')
ax.plot(rmid[okc], c_fit['ladder'][okc], '-', color='tab:blue', lw=2, label='ladder')
ax.set_xlabel('r [arcsec]'); ax.set_ylabel('g - i  [mag]')
ax.set_title('color gradient', fontsize=11); ax.legend(fontsize=8)
fig.suptitle('bulge (n=4, red) + disk (exp, blue), s/n ~ %d: exp vs ladder fits' % fits['exp']['obj']['s2n'], fontsize=13)
fig.savefig(OUT, dpi=110, facecolor='white')
print('wrote', OUT)
