"""Model for notebook 01 (group formation).  Kept in a file so that the
notebook can import it and run replicate populations in parallel."""
import numpy as np
from protoevo import Chem, Colony, ancestral_state, sole_repressor_pairs, soft_activity


def setup_chemistry(N=30, seed=1, omega=0.16, dt=0.02):
    chem = Chem(N=N, seed=seed, omega=omega, dt=dt)
    anc = ancestral_state(chem)
    s0 = chem.state(anc)
    pairs = sole_repressor_pairs(chem, s0)
    # separase-like species: active in the ancestor, sole repressor of some j
    sep, sep_partner = pairs[0]
    # adhesin-like species: inactive in the ancestor, with a sole repressor
    adh = None
    for k, j in pairs[1:]:
        if j != sep_partner and k != sep and j != sep:
            adh, adh_rep = j, k
            break
    roles = dict(sep=sep, sep_partner=sep_partner, adh=adh, adh_rep=adh_rep)
    return chem, anc, s0, roles


def logistic(x, x0=0.5, w=0.06):
    return 1.0 / (1.0 + np.exp(-(x - x0) / w))


def evolve(chem, anc, roles, days=40, K=2000, keep_frac=0.1, g0=0.5,
           group_cost=0.2, selection='settle', route='clonal', D_link=0.5,
           S_max=32, agg_size=8, n_pred=6.0, seed=0, div_every=5,
           record_states=False, mu_tag=0.05):
    """Serial-transfer evolution experiment (Ratcliff et al. 2012 style).

    route = 'clonal'      : a cell whose separase species is OFF fails to
                            separate from its daughter (ACE2-like) -> snowflake
    route = 'aggregative' : a cell whose adhesin species is ON sticks to other
                            sticky cells at the end of each day
    selection = 'settle' | 'predation' | 'none'
    """
    rng = np.random.default_rng(seed)
    sep, adh = roles['sep'], roles['adh']
    P0 = soft_activity(anc)
    n0 = int(K * keep_frac)
    tag_counter = [n0]
    col = Colony(np.tile(anc, (n0, 1)), traits={'tag': np.arange(n0)})
    T_day = np.log(3 / keep_frac) / g0   # growth halts at K (stationary phase)
    nsteps = int(T_day / chem.dt)
    hist = {k: [] for k in ['frac_multi', 'mean_size', 'sep_off', 'adh_on',
                            'P', 'within_sim', 'relatedness', 'rel_background', 'n_cells']}
    for day in range(days):
        if route == 'aggregative':            # aggregates disperse each day
            col.par[:] = -1
            col.grp = np.arange(col.C)
            col._next_gid = col.C
            col._changed()
        for k in range(nsteps):
            if col.C >= K:          # stationary phase: stop growing for today
                break
            size, _, _, _ = col.group_sizes() if k % div_every == 0 else (size, 0, 0, 0)
            col.X = chem.step(col.X, rng, links=col.links(), D=D_link)
            if k % div_every == 0:
                g = g0 * soft_activity(col.X) / P0 * np.where(size > 1, 1 - group_cost, 1.0)
                col.mass *= np.exp(g * chem.dt * div_every)
                who = np.nonzero(col.mass >= 2)[0]
                if len(who):
                    if route == 'clonal':
                        p_att = 1 - logistic(col.X[who, sep])
                    else:
                        p_att = np.zeros(len(who))
                    new = col.divide(who, rng.random(len(who)) < p_att, rng)
                    # neutral 'microsatellite' marker: mutates at division
                    mut = new[rng.random(len(new)) < mu_tag]
                    col.traits['tag'][mut] = tag_counter[0] + np.arange(len(mut))
                    tag_counter[0] += len(mut)
                    # packing limit: very large snowflakes fracture
                    size, gids, cnt, _ = col.group_sizes()
                    for gid in gids[cnt > S_max]:
                        col.fracture(gid, rng)
        # ---- aggregation (aggregative route) --------------------------------
        if route == 'aggregative':
            sticky = np.nonzero(col.X[:, adh] > 0.5)[0]
            rng.shuffle(sticky)
            for b in range(0, len(sticky) - 1, agg_size):
                chunk = sticky[b:b + agg_size]
                if len(chunk) > 1:
                    col.merge_groups(col.grp[chunk], rng)
        # ---- record --------------------------------------------------------
        size, gids, cnt, inv = col.group_sizes()
        S = (col.X > 0.5).astype(np.int8)
        hist['frac_multi'].append(np.mean(size > 1))
        hist['mean_size'].append(np.mean(size))
        hist['sep_off'].append(np.mean(S[:, sep] == 0))
        hist['adh_on'].append(np.mean(S[:, adh] == 1))
        hist['P'].append(S.mean() if len(S) else np.nan)
        hist['n_cells'].append(col.C)
        # within-group compositional similarity (relatedness proxy)
        multi = gids[cnt > 1]
        sims = []
        for gid in multi[:200]:
            m = S[col.grp == gid]
            sims.append(np.mean([(m[a] == m[b]).mean() for a in range(len(m))
                                 for b in range(a + 1, len(m))]))
        hist['within_sim'].append(np.mean(sims) if sims else np.nan)
        # relatedness: P(two cells of a group carry the same neutral marker)
        rel = []
        tag = col.traits['tag']
        for gid in multi[:300]:
            t = tag[col.grp == gid]
            _, c = np.unique(t, return_counts=True)
            n = len(t)
            rel.append((np.sum(c * (c - 1))) / (n * (n - 1)))
        hist['relatedness'].append(np.mean(rel) if rel else np.nan)
        _, c = np.unique(tag, return_counts=True)
        hist['rel_background'].append(np.sum(c * (c - 1)) / (col.C * (col.C - 1)))
        # ---- selection & transfer ---------------------------------------------
        target = int(keep_frac * K)
        if selection == 'settle':
            v = cnt ** (2 / 3) * np.exp(0.25 * rng.standard_normal(len(cnt)))
            order = np.argsort(-v)
        elif selection == 'predation':
            p_surv = 0.1 + 0.9 * cnt ** 3 / (cnt ** 3 + n_pred ** 3)
            surv = rng.random(len(cnt)) < p_surv
            order = rng.permutation(np.nonzero(surv)[0])
        else:
            order = rng.permutation(len(cnt))
        cum = np.cumsum(cnt[order])
        chosen = gids[order[:np.searchsorted(cum, target) + 1]]
        col.keep_groups(chosen)
    out = {k: np.array(v) for k, v in hist.items()}
    out['T_day'] = T_day
    out['gen_per_day'] = np.log2(1 / keep_frac)
    if record_states:
        out['final_states'] = (col.X > 0.5).astype(np.int8)
        out['final_grp'] = col.grp.copy()
        out['final_par'] = col.par.copy()
    return out


def run_one(job):
    """One replicate population (for run_parallel)."""
    job = dict(job)
    chem, anc, s0, roles = setup_chemistry(omega=job.pop('omega', 0.14))
    return evolve(chem, anc, roles, **job)
