"""a random blend of bulge- and disk-dominated galaxies fit with exp and
ladder: color data and residuals, per-band residual significance maps,
and per-object recovery of flux, color and fit quality"""
import sys, numpy as np
sys.path.insert(0, '/home/esheldon/git/kdeblend/tests')
from _sims import make_blend_mbobs, SCALE
from kdeblend import deblend
from kdeblend.render import render_model
from kdeblend.vis import make_color_image
from kdeblend.deblender import build_deblender
from ngmix.prepsfadmom import get_phase_angles
from ngmix.prepsfadmom.prepsfadmom_nb import admom_ksums
from ngmix.prepsfadmom.models import cov_from_e
from scipy.ndimage import gaussian_filter
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

OUT = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 23
rng = np.random.RandomState(seed)
dim = 160; psf_fwhms = [0.9, 0.85, 0.8]; noise = 0.9
BULGE_COLOR = np.array([0.25, 0.6, 1.0]); DISK_COLOR = np.array([0.7, 0.85, 1.0])

# random scene: ngal galaxies within +-5 arcsec, at least 1.4 arcsec apart
ngal = 8
pos = []
while len(pos) < ngal:
    p = rng.uniform(-5, 5, size=2)
    if all(np.hypot(*(p - q)) > 1.4 for q in pos):
        pos.append(p)
GALS = []
for k, (v, u) in enumerate(pos):
    fi = 10 ** rng.uniform(np.log10(120), np.log10(3000))
    bt = rng.choice([0.05, 0.15, 0.3, 0.5, 0.7, 0.85])
    hb = rng.uniform(0.25, 0.8); hd = rng.uniform(0.4, 1.8)
    e = rng.uniform(-0.3, 0.3, size=2)
    tint = rng.uniform(0.85, 1.15)      # per-galaxy color variation
    GALS.append((k + 1, (v, u), fi, bt, hb, hd, tuple(e), tint))
def comps_of(gal):
    k, (v, u), fi, bt, hb, hd, (e1, e2), tint = gal
    bc = BULGE_COLOR ** tint; dc = DISK_COLOR ** tint
    return [[dict(kind='sersic', n=4.0, hlr=hb, flux=fi * bt * bc[b], e1=e1 * 0.5, e2=e2 * 0.5, v=v, u=u),
             dict(kind='exp', hlr=hd, flux=fi * (1 - bt) * dc[b], e1=e1, e2=e2, v=v, u=u)] for b in range(3)]
scene = [sum((comps_of(g)[b] for g in GALS), []) for b in range(3)]
mbobs = make_blend_mbobs(scene, psf_fwhms, noise=noise, rng=rng, dim=dim)
data = [mbobs[b][0].image for b in range(3)]
true_flux = np.array([[sum(c['flux'] for c in comps_of(g)[b]) for b in range(3)] for g in GALS])
true_gi = -2.5 * np.log10(true_flux[:, 0] / true_flux[:, 2])

# each object's detection footprint size (the driver passes sep's
# isophotal moments): the second moments of its own noiseless
# i-band image within 2 arcsec, so the footprint weight bound applies
cen = (dim - 1) / 2
yy, xx = np.mgrid[:dim, :dim]
def tdet_of(g):
    im = make_blend_mbobs(comps_of(g), psf_fwhms, dim=dim)[2][0].image
    v, u = g[1]
    dv = (yy - cen) * SCALE - v; du = (xx - cen) * SCALE - u
    m = (np.hypot(dv, du) < 2.0) & (im > 0)
    w = im[m]
    return float(((dv[m] ** 2 + du[m] ** 2) * w).sum() / w.sum())
objects = [dict(v=g[1][0], u=g[1][1], Tguess=0.5, Tdet=tdet_of(g)) for g in GALS]
print('Tdet per object:', [round(o['Tdet'], 2) for o in objects])
fits = {}
for t in ('exp', 'ladder'):
    res = deblend(mbobs, [dict(o, type=t) for o in objects], rng=np.random.RandomState(0), full_errors=(t == 'ladder'))
    model = [render_model(res['objects'], mbobs[b][0], b, Tsmooth=res['Tsmooth']) for b in range(3)]
    fits[t] = dict(res=res, model=model, objs=res['objects'])
    print(f"=== {t}: converged {res['converged']} in {res['numiter']} sweeps")
def flux_sum_under(ep, v, u, W):
    """the flux sum of one prepped epoch under weight W centered at (v, u)"""
    alpha, beta = get_phase_angles(ep, v - ep['vcen'], u - ep['ucen'])
    sums = np.zeros(6)
    admom_ksums(ep['kim'], ep['iy'], ep['ix'], ep['dim'], alpha, beta, ep['kv'], ep['ku'],
                W[0, 0], W[0, 1], W[1, 1], ep['df2'], sums)
    return sums[5] * ep['detAtinv']
# each object's own noiseless truth, prepped with the fit's smoothing
truth_eps = []
for g in GALS:
    tm = make_blend_mbobs(comps_of(g), psf_fwhms, dim=dim)
    tdeb, _ = build_deblender(tm, [dict(v=g[1][0], u=g[1][1], Tguess=0.5)], fwhm_smooth=fits['exp']['res']['fwhm_smooth'], rng=np.random.RandomState(0))
    truth_eps.append(tdeb.epochs_per_obj[0])
for t in fits:
    res = fits[t]['res']; sm = res['Tsmooth'] / 2
    for k, (g, o) in enumerate(zip(GALS, res['objects'])):
        Sw = cov_from_e(o['gauss_e1'], o['gauss_e2'], o['gauss_T']) + np.diag([sm, sm])
        fs_true = [flux_sum_under(truth_eps[k][b], g[1][0], g[1][1], Sw) for b in range(3)]
        o['true_ap_gi'] = -2.5 * np.log10(fs_true[0] / fs_true[2])
        o['ap_gi'] = -2.5 * np.log10(o['gauss_flux'][0] / o['gauss_flux'][2])
        C = o.get('gauss_flux_cov')
        if t == 'ladder' and C is not None and np.all(np.isfinite(C)):
            Fg, Fi = o['gauss_flux'][0], o['gauss_flux'][2]
            var = C[0, 0] / Fg ** 2 + C[2, 2] / Fi ** 2 - 2 * C[0, 2] / (Fg * Fi)
            o['ap_gi_err'] = (2.5 / np.log(10)) * np.sqrt(max(var, 0.0))
        else:
            o['ap_gi_err'] = np.nan
print(f"{'gal':>3s} {'B/T':>4s} {'flux_i':>7s} {'s2n':>5s} | {'exp flux%':>9s} {'lad ap%':>8s} {'lad tot%':>8s} {'+-tot':>6s} | {'exp dap':>8s} {'lad dap':>8s} {'+-':>6s} | {'chi2 exp':>8s} {'chi2 lad':>8s}")
rows = []
for k, g in enumerate(GALS):
    v, u = g[1]
    r = np.hypot((yy - cen) * SCALE - v, (xx - cen) * SCALE - u)
    m = r < 1.5
    oe, ol = fits['exp']['objs'][k], fits['ladder']['objs'][k]
    chi2 = {t: float((((data[2] - fits[t]['model'][2]) ** 2)[m] / noise ** 2).sum() / m.sum()) for t in fits}
    dgi = {t: fits[t]['objs'][k]['ap_gi'] - fits[t]['objs'][k]['true_ap_gi'] for t in fits}
    row = dict(k=g[0], bt=g[3], fi=true_flux[k, 2], s2n=ol['s2n'],
               exp_flux=(oe['flux'][2] / true_flux[k, 2] - 1) * 100,
               lad_ap=(ol['flux'][2] / true_flux[k, 2] - 1) * 100,
               lad_tot=(ol['total_flux'][2] / true_flux[k, 2] - 1) * 100,
               exp_dgi=dgi['exp'], lad_dgi=dgi['ladder'],
               lad_ap_err=ol['flux_err'][2] / true_flux[k, 2] * 100,
               lad_tot_err=ol['total_flux_err'][2] / true_flux[k, 2] * 100,
               lad_dgi_err=ol['ap_gi_err'],
               chi2_exp=chi2['exp'], chi2_lad=chi2['ladder'], flags=(oe['deblend_flags'], ol['deblend_flags']))
    rows.append(row)
    print(f"{row['k']:3d} {row['bt']:4.2f} {row['fi']:7.0f} {row['s2n']:5.0f} | {row['exp_flux']:+8.1f} {row['lad_ap']:+8.1f} {row['lad_tot']:+8.1f} {row['lad_tot_err']:6.1f} | {row['exp_dgi']:+8.3f} {row['lad_dgi']:+8.3f} {row['lad_dgi_err']:6.3f} | {row['chi2_exp']:8.2f} {row['chi2_lad']:8.2f}  flags {row['flags']}")

# figure
fig = plt.figure(figsize=(16, 13), facecolor='white')
gs0 = fig.add_gridspec(4, 1, height_ratios=[1.25, 0.85, 0.85, 0.95], hspace=0.3, left=0.07, right=0.98, top=0.93, bottom=0.06)
gs_top = gs0[0].subgridspec(1, 3, wspace=0.05)
stretch = 4.0 * noise
for k, (title, ims) in enumerate([('data', data), ('data - exp', [d - m for d, m in zip(data, fits['exp']['model'])]),
                                  ('data - ladder', [d - m for d, m in zip(data, fits['ladder']['model'])])]):
    ax = fig.add_subplot(gs_top[0, k])
    ax.imshow(make_color_image(ims, stretch=stretch), origin='lower', extent=[-cen * SCALE, cen * SCALE] * 2)
    if k == 0:
        for g in GALS:
            ax.text(g[1][1] + 0.4, g[1][0] + 0.4, str(g[0]), color='white', fontsize=9)
    ax.set_title(title, fontsize=12); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
for row_i, t in enumerate(('exp', 'ladder')):
    gs_r = gs0[1 + row_i].subgridspec(1, 3, wspace=0.05)
    for b in range(3):
        ax = fig.add_subplot(gs_r[0, b])
        sig = gaussian_filter((data[b] - fits[t]['model'][b]) / noise, 1.0) / (1.0 / (2 * np.sqrt(np.pi)))
        im = ax.imshow(sig, origin='lower', cmap='RdBu_r', vmin=-6, vmax=6, extent=[-cen * SCALE, cen * SCALE] * 2)
        ax.set_xlim(-8, 8); ax.set_ylim(-8, 8); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"(data - {t}) / noise, {'gri'[b]} band, smoothed 1 px", fontsize=10)
        if b == 2:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, ticks=[-5, 0, 5]); cb.set_label('sigma')
gs_b = gs0[3].subgridspec(1, 3, wspace=0.42)
fi = np.array([r['fi'] for r in rows]); bt = np.array([r['bt'] for r in rows]); s2n = np.array([r['s2n'] for r in rows])
ax = fig.add_subplot(gs_b[0, 0])
ax.axhline(0, color='k', lw=1)
ax.plot(s2n, [r['exp_flux'] for r in rows], 's', color='tab:red', ms=7, label='exp model flux')
ax.errorbar(s2n, [r['lad_ap'] for r in rows], yerr=[r['lad_ap_err'] for r in rows], fmt='v', color='lightsteelblue', ms=7, capsize=2, label='ladder adaptive-aperture flux')
ax.errorbar(s2n, [r['lad_tot'] for r in rows], yerr=[r['lad_tot_err'] for r in rows], fmt='o', color='tab:blue', ms=7, capsize=2, label='ladder total_flux')
for r in rows:
    if -60 < r['exp_flux'] < 100:
        ax.text(r['s2n'] * 1.06, r['exp_flux'], str(r['k']), fontsize=7, color='tab:red', clip_on=True)
ax.set_ylim(-60, 100)
for r in rows:
    if r['exp_flux'] > 100:
        ax.annotate(f"{r['k']}: exp {r['exp_flux']:+.0f}\\%", (r['s2n'], 95), color='tab:red', fontsize=8, ha='center', va='top',
                    arrowprops=dict(arrowstyle='->', color='tab:red'), xytext=(r['s2n'] * 1.8, 80))
ax.set_xscale('log'); ax.set_xticks([10, 30, 50, 100, 200]); ax.set_xticklabels(['10', '30', '50', '100', '200']); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel('s/n'); ax.set_ylabel('(flux - true) / true [\\%], i band')
ax.set_title('flux recovery', fontsize=11); ax.legend(fontsize=8, loc='lower right')
ax = fig.add_subplot(gs_b[0, 1])
ax.axhline(0, color='k', lw=1)
ax.plot(bt + rng.uniform(-0.01, 0.01, ngal), [r['exp_dgi'] for r in rows], 's', color='tab:red', ms=7, label='exp')
ax.errorbar(bt + rng.uniform(-0.01, 0.01, ngal), [r['lad_dgi'] for r in rows], yerr=[r['lad_dgi_err'] for r in rows], fmt='o', color='tab:blue', ms=7, capsize=2, label='ladder')
for r in rows:
    ax.text(r['bt'] + 0.015, r['lad_dgi'], str(r['k']), fontsize=7, color='tab:blue')
ax.set_ylim(-0.35, 0.35)
ax.set_xlabel('true bulge fraction'); ax.set_ylabel('fitted - true aperture g - i [mag]')
ax.set_title('color recovery: fitted weight applied to the object\'s own truth', fontsize=10); ax.legend(fontsize=8)
ax = fig.add_subplot(gs_b[0, 2])
ax.axhline(1, color='k', lw=1)
ax.plot(s2n, [r['chi2_exp'] for r in rows], 's', color='tab:red', ms=7, label='exp')
ax.plot(s2n, [r['chi2_lad'] for r in rows], 'o', color='tab:blue', ms=7, label='ladder')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks([10, 30, 50, 100, 200]); ax.set_xticklabels(['10', '30', '50', '100', '200'])
ax.set_yticks([1, 2, 3, 5]); ax.set_yticklabels(['1', '2', '3', '5']); ax.minorticks_off()
ax.set_xlabel('s/n'); ax.set_ylabel('reduced chi2 within 1.5 arcsec, i band')
ax.set_title('fit quality per object', fontsize=11); ax.legend(fontsize=8)
fig.suptitle(f"random blend of {ngal} galaxies (seed {seed}), red n=4 bulges + blue exp disks: exp (converged {fits['exp']['res']['converged']}, {fits['exp']['res']['numiter']} sweeps) vs ladder (converged {fits['ladder']['res']['converged']}, {fits['ladder']['res']['numiter']} sweeps)", fontsize=12)
fig.savefig(OUT, dpi=100, facecolor='white', bbox_inches='tight', pad_inches=0.3)
print('wrote', OUT)
