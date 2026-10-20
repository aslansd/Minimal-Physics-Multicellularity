"""Models for notebook 02 (the Darwinian threshold)."""
import numpy as np
from protoevo import (Chem, Colony, ancestral_state, sole_repressor_pairs,
                      soft_activity)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def robust_active_node(chem, s0):
    """An active species whose dip does NOT flip the network: every inactive
    neighbour has >= 2 active repressors.  Its concentration then fluctuates
    (demographic noise) without causing a state switch -> a graded, noisy
    cell-level trait."""
    for k in np.nonzero(s0 == 1)[0]:
        ok = all(sum(s0[m] for m in chem.nbrs[j]) >= 2 for j in chem.nbrs[k])
        if ok:
            return int(k)
    raise RuntimeError('no robust node')


# ---------------------------------------------------------------------------
# A. Biophysical scaffolding: snowflake growth + packing-driven fracture
# ---------------------------------------------------------------------------

def snowflake_lineages(chem, anc, elong, Lam_e, n_groups=40, T=60.0, g0=0.6,
                       kappa=1.0, n0=6.0, beta=3.0, sigma_frac=0.08,
                       omega=0.25, seed=0, D_link=0.0, max_cells=4000):
    """Grow snowflake clusters of one genotype (supply Lam_e of the
    elongation species) and let them fracture under packing stress.

    Cell aspect ratio is fixed at birth:  AR = 1 + kappa * X_elong.
    A cluster of n cells fractures when n exceeds
        n* = n0 * <AR>^beta * exp(sigma_frac * xi)        (xi ~ N(0,1) per cluster)
    i.e. elongated cells pack loosely so clusters grow larger before the
    internal strain exceeds bond strength (Jacobeen et al. 2018).
    Fracture severs a random mother-daughter bond -> a clonal branch
    (propagule) is released.  Returns per-cell AR at birth and cluster sizes
    at fracture, plus genealogical info for parent-offspring regression.
    """
    rng = np.random.default_rng(seed)
    Lam = np.ones(chem.N)
    Lam[elong] = Lam_e
    X0 = chem.relax(anc[None, :], T=10, Lam=Lam)[0]
    col = Colony(np.tile(X0, (n_groups, 1)),
                 traits={'AR': 1 + kappa * np.full(n_groups, X0[elong])})
    thr = {g: n0 * np.exp(sigma_frac * rng.standard_normal()) for g in col.grp}
    parent_mean_AR = {g: np.nan for g in col.grp}
    parent_size = {g: np.nan for g in col.grp}
    rec_AR, rec_frac, rec_par_size, rec_meanAR = [], [], [], []
    size_trace = []
    nsteps, every = int(T / chem.dt), 10
    for k in range(nsteps):
        col.X = chem.step(col.X, rng, Lam=Lam, omega=omega,
                          links=col.links() if D_link else None, D=D_link)
        if k % every:
            continue
        col.mass *= np.exp(g0 * chem.dt * every)
        who = np.nonzero(col.mass >= 2)[0]
        if len(who):
            new = col.divide(who, np.ones(len(who), bool), rng)
            col.traits['AR'][new] = 1 + kappa * col.X[new, elong]
            rec_AR.extend(col.traits['AR'][new])
        size, gids, cnt, inv = col.group_sizes()
        size_trace.append(size[0])
        meanAR = np.bincount(inv, col.traits['AR']) / cnt
        for gi, gid in enumerate(gids):
            nstar = thr[gid] * meanAR[gi] ** beta
            if cnt[gi] > nstar:
                rec_frac.append(cnt[gi])
                rec_meanAR.append(meanAR[gi])
                rec_par_size.append(parent_size[gid])
                branch = col.fracture(gid, rng)
                newg = col.grp[branch[0]]
                # the propagule inherits nothing but its cells; new bond noise
                thr[newg] = n0 * np.exp(sigma_frac * rng.standard_normal())
                parent_size[newg] = cnt[gi]
                parent_size[gid] = cnt[gi]
        if col.C > max_cells:      # keep the census bounded: drop random groups
            _, gids, cnt, _ = col.group_sizes()
            keep = rng.choice(gids, size=max(1, len(gids) // 2), replace=False)
            col.keep_groups(keep)
    return dict(AR=np.array(rec_AR), frac_size=np.array(rec_frac),
                frac_meanAR=np.array(rec_meanAR),
                parent_size=np.array(rec_par_size),
                size_trace=np.array(size_trace))


def snowflake_evolution(chem, anc, elong, days=30, K=2000, keep_frac=0.1,
                        g0=0.6, kappa=1.0, n0=6.0, beta=3.0, sigma_frac=0.08,
                        omega=0.25, mu=0.02, mu_sd=0.15, selection='settle',
                        seed=0):
    """MuLTEE-like experiment: snowflake clusters, daily settling selection.
    The *genetic* parameter is Lam_e (catalytic supply of the elongation
    species), mutating at division; the selected trait is cluster size, a
    group-level property that emerges from cell shape via packing physics."""
    rng = np.random.default_rng(seed)
    n_start = int(K * keep_frac)
    col = Colony(np.tile(anc, (n_start, 1)),
                 traits={'Lam_e': np.ones(n_start),
                         'AR': 1 + kappa * np.full(n_start, anc[elong])})
    thr = {}
    T_day = np.log(3 / keep_frac) / g0
    hist = {k: [] for k in ['Lam_e', 'AR', 'size', 'n_frac_size']}
    every = 10
    for day in range(days):
        frac_sizes = []
        for k in range(int(T_day / chem.dt)):
            if col.C >= K:
                break
            Lam = np.ones((col.C, chem.N))
            Lam[:, elong] = col.traits['Lam_e']
            col.X = chem.step(col.X, rng, Lam=Lam, omega=omega)
            if k % every:
                continue
            col.mass *= np.exp(g0 * chem.dt * every)
            who = np.nonzero(col.mass >= 2)[0]
            if len(who) == 0:
                continue
            new = col.divide(who, np.ones(len(who), bool), rng)
            m = new[rng.random(len(new)) < mu]
            col.traits['Lam_e'][m] *= np.exp(mu_sd * rng.standard_normal(len(m)))
            col.traits['AR'][new] = 1 + kappa * col.X[new, elong]
            size, gids, cnt, inv = col.group_sizes()
            meanAR = np.bincount(inv, col.traits['AR']) / cnt
            for gi, gid in enumerate(gids):
                if gid not in thr:
                    thr[gid] = n0 * np.exp(sigma_frac * rng.standard_normal())
                if cnt[gi] > thr[gid] * meanAR[gi] ** beta:
                    frac_sizes.append(cnt[gi])
                    col.fracture(gid, rng)
        size, gids, cnt, inv = col.group_sizes()
        hist['Lam_e'].append(col.traits['Lam_e'].mean())
        hist['AR'].append(col.traits['AR'].mean())
        hist['size'].append(size.mean())
        hist['n_frac_size'].append(np.mean(frac_sizes) if frac_sizes else np.nan)
        target = int(keep_frac * K)
        if selection == 'settle':
            v = cnt ** (2 / 3) * np.exp(0.25 * rng.standard_normal(len(cnt)))
            order = np.argsort(-v)
        else:
            order = rng.permutation(len(cnt))
        cum = np.cumsum(cnt[order])
        col.keep_groups(gids[order[:np.searchsorted(cum, target) + 1]])
    return {k: np.array(v) for k, v in hist.items()}


def broad_sense_H2(values_by_genotype):
    """H^2 = Var(genotype means) / Var(all observations) (one-way ANOVA)."""
    allv = np.concatenate(values_by_genotype)
    means = np.array([v.mean() for v in values_by_genotype])
    ns = np.array([len(v) for v in values_by_genotype])
    vb = np.sum(ns * (means - allv.mean()) ** 2) / ns.sum()
    return vb / allv.var()


# ---------------------------------------------------------------------------
# B. Ecological scaffolding: patches founded by bottlenecks, flood dispersal
# ---------------------------------------------------------------------------

def scaffold_evolution(chem, anc, roles, cycles=60, n_patches=100,
                       founders=1, T_mat=6.0, g0=0.6, cost=0.3, b0=0.05,
                       omega=0.18, seed=0, well_mixed=False, n_cap=40):
    """Cells carry a compositional 'matrix' species m (ON = produces costly
    structural matrix).  Each cycle: patches on the cave wall are founded by
    `founders` cells drawn from the dispersal pool, cells grow (m-producers
    grow more slowly), then a flood releases dispersers; a patch releases
    dispersers in proportion to n_cells * (b0 + f_m) -- only mats held
    together by matrix survive the flood intact (Pseudomonas / J. sacculi).

    well_mixed=True removes the patch structure (no scaffold): one global
    population, dispersal success uses the GLOBAL matrix fraction, so there is
    no between-group variance for selection to act on.
    """
    rng = np.random.default_rng(seed)
    m = roles['adh']   # reuse: an inactive species with an accessible switch
    P0 = soft_activity(anc)
    pool = np.tile(anc, (n_patches * founders, 1))
    hist = {k: [] for k in ['f_m', 'cell_growth', 'group_output',
                            'between_var', 'n_cells']}
    po_parent, po_child = [], []
    parent_fm = None
    cap_total = n_cap * n_patches
    for cyc in range(cycles):
        # ---- found patches ----
        grp = (np.zeros(len(pool), int) if well_mixed
               else np.repeat(np.arange(n_patches), founders))
        col = Colony(pool, grp=grp)
        nsteps, every = int(T_mat / chem.dt), 10
        for k in range(nsteps):
            col.X = chem.step(col.X, rng, omega=omega)
            if k % every:
                continue
            on = col.X[:, m] > 0.5
            g = g0 * soft_activity(col.X) / P0 * np.where(on, 1 - cost, 1.0)
            size, gids, cnt, inv = col.group_sizes()
            cap = cap_total if well_mixed else n_cap
            g = g * np.clip(1 - size / cap, 0, 1)        # crowding
            col.mass *= np.exp(g * chem.dt * every)
            who = np.nonzero(col.mass >= 2)[0]
            if len(who):
                col.divide(who, np.zeros(len(who), bool), rng)
                col.grp[-len(who):] = col.grp[who]       # daughters stay in patch
        # ---- flood: dispersal ----
        on = (col.X[:, m] > 0.5).astype(float)
        size, gids, cnt, inv = col.group_sizes()
        fm = np.bincount(inv, on) / cnt
        if parent_fm is not None and not well_mixed and founders == 1:
            po_parent.append(parent_fm)
            po_child.append(fm)
        hist['f_m'].append(on.mean())
        hist['cell_growth'].append(np.mean(1 - cost * on))
        hist['group_output'].append((cnt * (b0 + fm)).sum() / n_patches)
        hist['between_var'].append(fm.var() if not well_mixed else 0.0)
        hist['n_cells'].append(col.C)
        # every cell's chance to become a founder = its patch's per-cell output
        w = (b0 + fm)[inv]
        pick = rng.choice(col.C, size=n_patches * founders, p=w / w.sum())
        parent_fm = fm[inv[pick]]
        pool = col.X[pick]
    out = {k: np.array(v) for k, v in hist.items()}
    out['po_parent'] = np.concatenate(po_parent) if po_parent else np.array([])
    out['po_child'] = np.concatenate(po_child) if po_child else np.array([])
    return out


# ---------------------------------------------------------------------------
# C. Choanoeca flexa: collective inversion from independent cell switches
# ---------------------------------------------------------------------------

def flexa_sheet(darkness, n_cells=60, n_sheets=200, T=40.0, omega=0.2,
                f_crit=0.5, feedback=0.0, seed=0, dt=0.01):
    """Each cell carries a 2-species reciprocal-inhibition switch (paper Fig.
    1 with N=2): 'relaxed collar' R vs 'contracted collar' K.  Darkness
    (0..1) shifts catalytic supply from R to K (rhodopsin-cGMP pathway).
    Cells do NOT communicate.  The sheet is a mechanical continuum: it snaps
    to the inverted (swimming) configuration when the fraction of contracted
    collars exceeds f_crit.  Optional `feedback`: once inverted, the imposed
    curvature adds a small bias toward contraction in every cell.
    Returns fraction contracted per cell-time and inversion state per sheet.
    """
    import networkx as nx
    chem = Chem(graph=nx.path_graph(2), omega=omega, dt=dt)
    rng = np.random.default_rng(seed)
    C = n_cells * n_sheets
    X = np.tile([1.0, 0.0], (C, 1))
    Lam = np.empty((C, 2))
    inv_state = np.zeros(n_sheets, bool)
    traj = []
    for k in range(int(T / dt)):
        d = darkness(k * dt) if callable(darkness) else darkness
        bias = feedback * np.repeat(inv_state, n_cells)
        Lam[:, 0] = 1.0 - 0.5 * d - bias
        Lam[:, 1] = 0.5 + 0.5 * d + bias
        X = chem.step(X, rng, Lam=Lam)
        if k % 20 == 0:
            contracted = (X[:, 1] > X[:, 0]).reshape(n_sheets, n_cells)
            fc = contracted.mean(1)
            # snap-through with hysteresis (bistable shell)
            inv_state = np.where(inv_state, fc > f_crit - 0.15, fc > f_crit)
            traj.append((k * dt, contracted.mean(), inv_state.mean(),
                         contracted[0].copy(), fc.copy()))
    return traj


# ---------------------------------------------------------------------------
# picklable job wrappers for run_parallel
# ---------------------------------------------------------------------------

def _snow_chem():
    chem = Chem(N=30, seed=1)
    anc = ancestral_state(chem)
    return chem, anc, robust_active_node(chem, chem.state(anc))


def h2_job(job):
    chem, anc, e = _snow_chem()
    return snowflake_lineages(chem, anc, e, job['Le'], n_groups=30, T=job['T'],
                              omega=job['omega'], seed=job['seed'])


def multee_job(job):
    chem, anc, e = _snow_chem()
    return snowflake_evolution(chem, anc, e, days=job['days'], selection=job['selection'],
                               mu=0.1, mu_sd=0.1, seed=job['seed'])


def scaffold_job(job):
    from nb1_model import setup_chemistry
    chem, anc, s0, roles = setup_chemistry(omega=0.18)
    return scaffold_evolution(chem, anc, roles, cycles=job['cycles'],
                              founders=job['founders'], well_mixed=job['well_mixed'],
                              seed=job['seed'])
