"""a blend of bulge- and disk-dominated galaxies of different colors,
sizes and brightnesses, fit with exp and ladder: color images and
residuals, and per-object profiles and color gradients against each
object's noiseless truth"""
import sys, numpy as np
sys.path.insert(0, '/home/esheldon/git/kdeblend/tests')
from _sims import make_blend_mbobs, SCALE
from kdeblend import deblend
from kdeblend.render import render_model
from kdeblend.vis import make_color_image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = sys.argv[1]
rng = np.random.RandomState(11)
dim = 144; psf_fwhms = [0.9, 0.85, 0.8]; noise = 0.9
BULGE_COLOR = np.array([0.25, 0.6, 1.0])     # g, r, i relative to i
DISK_COLOR = np.array([0.7, 0.85, 1.0])
# name, (v, u) arcsec, i flux, bulge fraction, bulge hlr, disk hlr, (e1, e2)
GALS = [
    ('1 bright red B/T 0.7', (0.0, 0.0), 3000.0, 0.7, 0.8, 1.6, (0.05, -0.05)),
    ('2 blue disk B/T 0.15', (2.0, 3.5), 1200.0, 0.15, 0.3, 1.5, (0.25, 0.10)),
    ('3 faint blue disk', (1.5, -2.5), 250.0, 0.05, 0.2, 0.5, (-0.1, 0.2)),
    ('4 red B/T 0.6', (-3.5, 2.0), 600.0, 0.6, 0.4, 0.9, (0.1, 0.15)),
    ('5 faint red B/T 0.85', (-3.0, -3.0), 180.0, 0.85, 0.3, 0.6, (0.0, 0.1)),
]
def comps_of(gal, scale=1.0):
    name, (v, u), fi, bt, hb, hd, (e1, e2) = gal
    out = []
    for b in range(3):
        fb = fi * bt * BULGE_COLOR[b] * scale; fd = fi * (1 - bt) * DISK_COLOR[b] * scale
        out.append([dict(kind='sersic', n=4.0, hlr=hb, flux=fb, e1=e1 * 0.5, e2=e2 * 0.5, v=v, u=u),
                    dict(kind='exp', hlr=hd, flux=fd, e1=e1, e2=e2, v=v, u=u)])
    return out
scene = [sum((comps_of(g)[b] for g in GALS), []) for b in range(3)]
mbobs = make_blend_mbobs(scene, psf_fwhms, noise=noise, rng=rng, dim=dim)
data = [mbobs[b][0].image for b in range(3)]
truth_obj = [[make_blend_mbobs(comps_of(g), psf_fwhms, dim=dim)[b][0].image for b in range(3)] for g in GALS]
true_flux = [[sum(c['flux'] for c in comps_of(g)[b]) for b in range(3)] for g in GALS]

objects = [dict(v=g[1][0], u=g[1][1], Tguess=0.5) for g in GALS]
fits = {}
for t in ('exp', 'ladder'):
    res = deblend(mbobs, [dict(o, type=t) for o in objects], rng=np.random.RandomState(0))
    per_obj = [[render_model([o], mbobs[b][0], b, Tsmooth=res['Tsmooth']) for b in range(3)] for o in res['objects']]
    model = [sum(m[b] for m in per_obj) for b in range(3)]
    fits[t] = dict(res=res, per_obj=per_obj, model=model)
    print(f"=== {t}: converged {res['converged']} in {res['numiter']} sweeps, nskip {res['nskip']}")
    for k, (g, o) in enumerate(zip(GALS, res['objects'])):
        tf = true_flux[k]
        gi_true = -2.5 * np.log10(tf[0] / tf[2]); gi = -2.5 * np.log10(o['flux'][0] / o['flux'][2])
        line = f"  {g[0]:22s} s2n {o['s2n']:6.1f} T {o['T']:.3f} flux_i {o['flux'][2]:7.1f} (true {tf[2]:.0f}, {(o['flux'][2]/tf[2]-1)*100:+.1f}%)  g-i {gi:.3f} (true {gi_true:.3f}, {gi-gi_true:+.3f}) flags {o['deblend_flags']}"
        if t == 'ladder':
            line += f"  total_i {o['total_flux'][2]:.1f} ({(o['total_flux'][2]/tf[2]-1)*100:+.1f}%) gradient g-r {o['gradient'][0]:+.3f}"
        print(line)

cen = (dim - 1) / 2
yy, xx = np.mgrid[:dim, :dim]
edges = np.arange(0, 3.0 + 1e-9, 0.25); rmid = 0.5 * (edges[1:] + edges[:-1])
def prof(im, v, u):
    r = np.hypot((yy - cen) * SCALE - v, (xx - cen) * SCALE - u)
    return np.array([im[(r >= lo) & (r < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])])
def color_of(ims, v, u):
    with np.errstate(invalid='ignore', divide='ignore'):
        return -2.5 * np.log10(prof(ims[0], v, u) / prof(ims[2], v, u))

stretch = 4.0 * noise
fig = plt.figure(figsize=(16, 12.5), facecolor='white')
gs0 = fig.add_gridspec(3, 1, height_ratios=[1.15, 0.9, 0.9], hspace=0.3, left=0.05, right=0.99, top=0.94, bottom=0.05)
gs_top = gs0[0].subgridspec(1, 5, wspace=0.06)
gs_p = gs0[1].subgridspec(1, 5, wspace=0.28)
gs_c = gs0[2].subgridspec(1, 5, wspace=0.28)
half = cen * SCALE
panels = [('data', data), ('exp model', fits['exp']['model']), ('data - exp', [d - m for d, m in zip(data, fits['exp']['model'])]),
          ('ladder model', fits['ladder']['model']), ('data - ladder', [d - m for d, m in zip(data, fits['ladder']['model'])])]
for k, (title, ims) in enumerate(panels):
    ax = fig.add_subplot(gs_top[0, k])
    ax.imshow(make_color_image(ims, stretch=stretch), origin='lower', extent=[-half, half, -half, half])
    if k == 0:
        for g in GALS:
            ax.text(g[1][1] + 0.6, g[1][0] + 0.6, g[0].split()[0], color='white', fontsize=9)
    ax.set_title(title, fontsize=12); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-7, 7); ax.set_ylim(-7, 7)
for k, g in enumerate(GALS):
    v, u = g[1]
    p_true = prof(truth_obj[k][2], v, u)
    ax = fig.add_subplot(gs_p[0, k])
    ax.plot(rmid, p_true, 'k-', lw=2, label='truth (this object)')
    for t, c, ls in (('exp', 'tab:red', '--'), ('ladder', 'tab:blue', '-')):
        others = sum(m[2] for j, m in enumerate(fits[t]['per_obj']) if j != k)
        ax.plot(rmid, prof(data[2] - others, v, u), 'o', color=c, ms=3, alpha=0.6, label=f'data - other {t} models')
        ax.plot(rmid, prof(fits[t]['per_obj'][k][2], v, u), ls, color=c, lw=2, label=f'{t} model')
    ax.set_yscale('log'); ax.set_ylim(noise * 0.2, max(p_true.max() * 1.6, noise))
    ax.axhline(noise, color='0.6', ls=':', lw=1)
    ax.set_title(f"{g[0]}  (s/n {fits['exp']['res']['objects'][k]['s2n']:.0f})", fontsize=10)
    ax.set_xlabel('r [arcsec]')
    if k == 0:
        ax.set_ylabel('i-band SB [per pixel]'); ax.legend(fontsize=7, loc='lower left')
    ax = fig.add_subplot(gs_c[0, k])
    okc = p_true > 1.5 * noise
    ax.plot(rmid[okc], color_of(truth_obj[k], v, u)[okc], 'k-', lw=2, label='truth')
    for t, c, ls in (('exp', 'tab:red', '--'), ('ladder', 'tab:blue', '-')):
        ax.plot(rmid[okc], color_of(fits[t]['per_obj'][k], v, u)[okc], ls, color=c, lw=2, label=t)
    ax.set_ylim(0.3, 1.7); ax.set_xlabel('r [arcsec]')
    if k == 0:
        ax.set_ylabel('g - i  [mag]'); ax.legend(fontsize=8)
    ax.set_title('color gradient', fontsize=10)
fig.suptitle('a blend of five galaxies (red n=4 bulges, blue exp disks): exp vs ladder fits at the true positions', fontsize=13)
fig.savefig(OUT, dpi=105, facecolor='white')
print('wrote', OUT)
