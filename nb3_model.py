"""Model for notebook 03 (functional integration: germ-soma differentiation).

The same chemistry as before, with ONE environmental input: the catalytic
supply of a "nutrient-sensing" species q scales with the local nutrient level
u (Lam_q = u).  q is the sole repressor of a "stress" species r (RLS1-like).
When u drops, q falls below its binder threshold, r is de-repressed, switches
ON and in turn represses q: a bistable, environmentally induced cell state.
A cell with r ON stops dividing (-reproduction) but is protected and can
power group functions such as motility (+survival) -> the soma phenotype.
The genetic parameter theta = Lam_r (catalytic efficiency of the r-reaction)
sets the nutrient level at which the switch flips.
"""
import numpy as np
from protoevo import Chem, Colony, ancestral_state, sole_repressor_pairs


def setup(N=20, seed=3, omega=0.1, dt=0.02):
    chem = Chem(N=N, seed=seed, omega=omega, dt=dt)
    anc = ancestral_state(chem)
    s0 = chem.state(anc)
    q, r = sole_repressor_pairs(chem, s0)[0]
    return chem, anc, dict(q=q, r=r)


def lam_matrix(chem, roles, u, theta):
    u = np.atleast_1d(u)
    theta = np.atleast_1d(theta)
    C = max(len(u), len(theta))
    Lam = np.ones((C, chem.N))
    Lam[:, roles['q']] = u
    Lam[:, roles['r']] = theta
    return Lam


def reaction_norm(chem, anc, roles, u_values, theta, n_cells=200, T=30.0, seed=0):
    """Fraction of single cells that end in the r-ON (soma-like) state after
    exposure to nutrient level u for time T."""
    rng = np.random.default_rng(seed)
    out = []
    for u in u_values:
        X = np.tile(anc, (n_cells, 1))
        Lam = lam_matrix(chem, roles, np.full(n_cells, u), theta)
        X = chem.run(X, T, rng, Lam=Lam)
        out.append(np.mean(X[:, roles['r']] > 0.5))
    return np.array(out)


def nutrient_profile(q, n, u_ext, lam=1.0):
    """Steady-state diffusion-consumption profile inside a spherical cluster of
    n cells (radius R = n^(1/3) cell radii); q in (0,1) is a cell's radial
    quantile (rho = R q^(1/3)).  u(rho) = u_ext * R sinh(rho/l) / (rho sinh(R/l))."""
    R = np.maximum(n, 1) ** (1 / 3)
    rho = np.maximum(R * q ** (1 / 3), 1e-6)
    return u_ext * (R * np.sinh(rho / lam)) / (rho * np.sinh(R / lam))


def survival(f_soma, s0=0.25, a=0.1):
    """Group-level benefit of somatic (motile / protective) cells."""
    return s0 + (1 - s0) * f_soma / (f_soma + a)


def develop(chem, roles, X0, theta0, gid0, rng, T_dev=6.0, u_ext=1.0, lam=1.0,
            g0=1.0, mu=0.1, mu_sd=0.08, n_max=32, every=10, unicellular=False,
            t_mature=2.0, T_max=10.0):
    """Grow groups from founder cells (gid0 = group of each founder).
    Germ cells (r OFF) grow at rate g0*u_local and divide; daughters stay in
    the group (clonal development).  Soma cells (r ON) do not divide.
    A group is scored `t_mature` time units after it first reaches n_max
    cells (so groups founded by one or by several cells are compared at the
    same developmental stage); the cell fates at that moment are stored in
    col.traits['soma'].  With unicellular=True, cells grow independently for
    T_dev and fates are read at the end."""
    C0 = len(X0)
    col = Colony(X0, grp=gid0, traits={'theta': np.asarray(theta0, float).copy(),
                                       'q': rng.random(C0), 'soma': np.zeros(C0, bool)})
    col.mass = 1 + rng.random(C0)
    r = roles['r']
    ng = int(np.max(gid0)) + 1
    reached = np.full(ng, -1.0)
    done = np.zeros(ng, bool)
    T_end = T_dev if unicellular else T_max
    for k in range(int(T_end / chem.dt)):
        t = k * chem.dt
        size, gids, cnt, inv = col.group_sizes()
        u = (np.full(col.C, u_ext) if unicellular
             else nutrient_profile(col.traits['q'], size, u_ext, lam))
        Lam = lam_matrix(chem, roles, u, col.traits['theta'])
        col.X = chem.step(col.X, rng, Lam=Lam)
        if not unicellular:
            newly = (cnt >= n_max) & (reached[gids] < 0)
            reached[gids[newly]] = t
            ready = (reached[gids] >= 0) & (t - reached[gids] >= t_mature) & ~done[gids]
            for g in gids[ready]:
                m = col.grp == g
                col.traits['soma'][m] = col.X[m, r] > 0.5
                done[g] = True
            if done[gids].all():
                break
        if k % every:
            continue
        germ = col.X[:, r] < 0.5
        grow = germ & (size < n_max)
        col.mass[grow] *= np.exp(g0 * u[grow] * chem.dt * every)
        who = np.nonzero(col.mass >= 2)[0]
        if len(who):
            new = col.divide(who, np.zeros(len(who), bool), rng)
            col.grp[new] = col.grp[who]          # same body (no bonds needed)
            m = new[rng.random(len(new)) < mu]
            col.traits['theta'][m] = np.maximum(
                col.traits['theta'][m] + mu_sd * rng.standard_normal(len(m)), 0)
            col.traits['q'][new] = rng.random(len(new))
    # groups (or free cells) not yet scored: read fates now
    left = ~done[col.grp] if not unicellular else np.ones(col.C, bool)
    col.traits['soma'][left] = col.X[left, r] > 0.5
    return col


def evolve(chem, anc, roles, gens=80, G=100, mode='clonal', k_chim=4,
           theta0=0.6, u_ext=1.0, lam=1.0, T_dev=6.0, U=0, u_uni=0.45,
           T_uni=3.0, T_disp=3.0, seed=0, survival_kw=None, verbose=False):
    """Group life cycle: founders -> development -> group selection ->
    propagules.  mode='clonal': each group from ONE germ cell;
    mode='chimeric': each group aggregated from k_chim germ cells drawn from
    different parent groups.  U>0: facultative life cycle with U rounds of
    unicellular growth (mild stress u_uni) between group phases."""
    rng = np.random.default_rng(seed)
    survival_kw = survival_kw or {}
    k = 1 if mode == 'clonal' else k_chim
    X = np.tile(anc, (G * k, 1))
    theta = (np.full(G * k, theta0) if np.isscalar(theta0)
             else np.random.default_rng(seed + 1).choice(theta0, G * k))
    hist = {key: [] for key in ['theta', 'theta_sd', 'f_soma', 'W', 'n_cells',
                                'within_var', 'theta_uni']}
    r = roles['r']
    for gen in range(gens):
        # dispersal: propagules spend T_disp as free cells in the external medium
        Lam_d = lam_matrix(chem, roles, np.full(len(X), u_ext), theta)
        X = chem.run(X, T_disp, rng, Lam=Lam_d)
        gid = np.repeat(np.arange(G), k)
        col = develop(chem, roles, X, theta, gid, rng, T_dev=T_dev, u_ext=u_ext,
                      lam=lam)
        size, gids, cnt, inv = col.group_sizes()
        soma = col.traits['soma'].astype(float)
        f = np.bincount(inv, soma) / cnt
        n_germ = cnt * (1 - f)
        W = n_germ * survival(f, **survival_kw)
        th = col.traits['theta']
        gm = np.bincount(inv, th) / cnt
        hist['theta'].append(th.mean())
        hist['theta_sd'].append(th.std())
        hist['f_soma'].append(f.mean())
        hist['W'].append(W.mean())
        hist['n_cells'].append(cnt.mean())
        hist['within_var'].append(np.mean(np.bincount(inv, (th - gm[inv]) ** 2) / cnt))
        # ---- propagules: germ cells, weighted by their group's survival ----
        # propagules: uncommitted germ cells (stress species still below its binder level)
        germ = np.nonzero((soma == 0) & (col.X[:, r] < chem.B))[0]
        if len(germ) == 0:
            germ = np.nonzero(soma == 0)[0]
        w = survival(f, **survival_kw)[inv[germ]]
        pick = germ[rng.choice(len(germ), size=G * k, p=w / w.sum())]
        X, theta = col.X[pick].copy(), th[pick].copy()
        # founders start from the dividing (germ) state
        # ---- optional unicellular phase (facultative multicellularity) ----
        for _ in range(U):
            uc = develop(chem, roles, X, theta, np.arange(len(X)), rng,
                         T_dev=T_uni, u_ext=u_uni, unicellular=True)
            idx = rng.choice(uc.C, size=G * k, replace=False)
            X, theta = uc.X[idx].copy(), uc.traits['theta'][idx].copy()
            # cells that are soma-like at the end of the free-living phase
            # are quiescent spores: they re-enter the group phase as germ cells
            X[:, r] = 0.0
        hist['theta_uni'].append(theta.mean())
        if verbose and gen % 10 == 0:
            print(gen, round(th.mean(), 3), round(f.mean(), 3), round(W.mean(), 2))
    return {key: np.array(v) for key, v in hist.items()}


def evolve_job(job):
    """Picklable wrapper for run_parallel."""
    chem, anc, roles = setup()
    return evolve(chem, anc, roles, **job)
