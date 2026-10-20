"""
protoevo.py -- shared simulation engine
=======================================

Extends the coarse-grained model of Bunin & Rivoire ("Evolutionary features in a
minimal physical system", arXiv:2411.03843) from a single reactor / 1-D tube to
*populations of compartments* ("protocells") that grow, divide, stay attached,
fragment and are selected.  Everything the notebooks do is built on one equation
(paper Eq. 1, with the demographic noise of Eq. C16 and diffusion of Eq. D3):

    dA_i = [ Lam_i / (1 + Gamma * sum_{j~i} max(0, A_j - B_j)) - delta*A_i ] dt
           + omega * sqrt(c_i + delta*A_i) dW_i
           + (diffusion between compartments that are physically linked)

* the sparse, reciprocal inhibition graph  i~j  is the "chemistry" (fixed);
* a stable state = maximal independent set of that graph = a heritable
  compositional "genotype" (paper Sec. III);
* noise-driven jumps between stable states = "mutations";
* diffusion between linked compartments = cells sharing cytoplasm / ECM.

The only new ingredients added here are physical book-keeping rules for
compartments (growth, division, attachment, fragmentation), each of which is a
*read-out* of the chemistry (e.g. a cell separates from its daughter only if a
"separase" species is active).

Pure numpy / scipy / networkx -- runs on a laptop.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import networkx as nx

# --------------------------------------------------------------------------
# 1. The chemistry: inhibition graph + parameters
# --------------------------------------------------------------------------


class Chem:
    """Inhibition network and kinetic parameters of the coarse-grained model.

    Parameters follow the paper's defaults: Gamma=10, B=0.25, delta=1, Lam=1.
    `omega` is the demographic-noise amplitude of ONE compartment
    (omega = Omega^{-1/2}, Omega = compartment volume).
    """

    def __init__(self, N=50, c=3, seed=0, Gamma=10.0, B=0.25, delta=1.0,
                 omega=0.1, dt=0.01, graph=None):
        if graph is None:
            graph = nx.random_regular_graph(c, N, seed=seed)
        self.G = nx.convert_node_labels_to_integers(graph)
        self.N = self.G.number_of_nodes()
        self.A = nx.to_numpy_array(self.G, nodelist=range(self.N))
        self.nbrs = [list(self.G.neighbors(i)) for i in range(self.N)]
        self.Gamma, self.B, self.delta = Gamma, B, delta
        self.omega, self.dt = omega, dt

    # ---- rates ---------------------------------------------------------
    def production(self, X, Lam=1.0, B=None):
        """Catalysed production rate c_i of every species in every compartment."""
        B = self.B if B is None else B
        inh = np.maximum(X - B, 0.0) @ self.A
        return Lam / (1.0 + self.Gamma * inh)

    # ---- one Euler-Maruyama step (paper Appendix F, Eq. F1) -------------
    def step(self, X, rng, Lam=1.0, B=None, omega=None, links=None, D=0.0,
             noise=True):
        """Advance concentrations X (C x N) by one time step dt.

        links : Links object (compartments that exchange molecules), or None
        omega : scalar or (C,1) array of per-compartment noise amplitudes
        """
        dt = self.dt
        omega = self.omega if omega is None else omega
        c = self.production(X, Lam, B)
        dX = (c - self.delta * X) * dt
        if noise:
            dX += omega * np.sqrt((c + self.delta * X) * dt) * \
                rng.standard_normal(X.shape)
        if links is not None and D > 0 and links.E > 0:
            dX += links.diffuse(X, D, dt, omega if noise else 0.0, rng)
        return np.maximum(X + dX, 0.0)

    def run(self, X, T, rng, record_every=None, **kw):
        nsteps = int(round(T / self.dt))
        rec = []
        for k in range(nsteps):
            X = self.step(X, rng, **kw)
            if record_every and k % record_every == 0:
                rec.append(X.copy())
        return (X, np.array(rec)) if record_every else X

    def relax(self, X, T=30.0, **kw):
        """Deterministic relaxation to the nearest stable state (Appendix F.3)."""
        rng = np.random.default_rng(0)
        kw = dict(kw)
        kw['noise'] = False
        return self.run(np.array(X, float), T, rng, **kw)

    # ---- states --------------------------------------------------------
    def state(self, X, Lam=1.0):
        """Binary state (active = 1). Threshold at half of Lam/delta."""
        thr = 0.5 * np.asarray(Lam) / self.delta
        return (X > thr).astype(np.int8)

    def stable_state(self, X, Lam=1.0, B=None, T=30.0):
        return self.state(self.relax(X, T=T, Lam=Lam, B=B), Lam)

    def is_stable(self, s):
        """s (binary, N) is a maximal independent set of the graph?"""
        s = np.asarray(s)
        nact = self.A @ s
        indep = np.all(nact[s == 1] == 0)
        maximal = np.all(nact[s == 0] >= 1)
        return bool(indep and maximal)

    def enumerate_states(self, max_states=2_000_000):
        """All maximal independent sets (= stable states), by Bron-Kerbosch on
        the complement graph.  Feasible up to N ~ 40 on a laptop."""
        Gc = nx.complement(self.G)
        out = []
        for k, cl in enumerate(nx.find_cliques(Gc)):
            v = np.zeros(self.N, np.int8)
            v[list(cl)] = 1
            out.append(v)
            if k + 1 >= max_states:
                break
        return np.array(out)

    def state_to_X(self, s, Lam=1.0, B=None, T=20.0):
        """Concentration vector sitting in stable state s."""
        X0 = np.asarray(s, float) * (np.ones(self.N) * Lam) / self.delta
        return self.relax(X0[None, :], T=T, Lam=Lam, B=B)[0]


def productivity(states):
    """Fraction of active species (paper's P)."""
    return np.asarray(states).mean(axis=-1)


def similarity(s1, s2):
    """Fraction of species with the same on/off status (paper Fig. 3C)."""
    return (np.asarray(s1) == np.asarray(s2)).mean(axis=-1)


def soft_activity(X, Lam=1.0, delta=1.0):
    """Smooth version of the active fraction: mean of clip(X / (Lam/delta))."""
    return np.clip(X * delta / Lam, 0, 1).mean(axis=-1)


# --------------------------------------------------------------------------
# 2. Physical links between compartments (diffusion, Appendix D)
# --------------------------------------------------------------------------


class Links:
    """Undirected links between compartments; molecules hop across links.

    Implements the discrete diffusion "reactions" A(x) -> A(x') of Appendix D,
    including their own Poisson (here Gaussian) noise, for an arbitrary graph
    of compartments (a 1-D tube, a snowflake tree, a sheet ...).
    """

    def __init__(self, edges, C):
        edges = np.asarray(edges, int).reshape(-1, 2)
        self.E, self.C = len(edges), C
        self.edges = edges
        if self.E:
            r = np.repeat(np.arange(self.E), 2)
            cidx = edges.ravel()
            val = np.tile([1.0, -1.0], self.E)
            self.inc = sp.csr_matrix((val, (r, cidx)), shape=(self.E, C))
            self.incT = self.inc.T.tocsr()
            self.absinc = abs(self.inc)

    def diffuse(self, X, D, dt, omega, rng):
        grad = self.inc @ X                      # X_u - X_v per link
        flux = D * grad * dt
        if np.any(omega):
            om = omega if np.isscalar(omega) else (self.absinc @ omega) / 2
            tot = self.absinc @ X                # X_u + X_v
            flux += om * np.sqrt(D * tot * dt) * rng.standard_normal(grad.shape)
        return -(self.incT @ flux)

    @staticmethod
    def ring(M, n_systems=1):
        """n independent 1-D periodic tubes of M cells (the paper's geometry)."""
        e = []
        for s in range(n_systems):
            o = s * M
            for m in range(M if M > 2 else M - 1):
                e.append((o + m, o + (m + 1) % M))
        return Links(e, M * n_systems)

    @staticmethod
    def chain(M, n_systems=1):
        e = [(s * M + m, s * M + m + 1) for s in range(n_systems)
             for m in range(M - 1)]
        return Links(e, M * n_systems)


# --------------------------------------------------------------------------
# 3. A population of compartments that grow, divide, attach and fragment
# --------------------------------------------------------------------------


class Colony:
    """Book-keeping for cells (compartments) organised in groups.

    Attributes (one entry per cell):
        X     : (C, N) concentrations -- the compositional "genome"
        grp   : group label;  cells with the same label form one body
        par   : index of the cell it is physically bonded to (-1 = none);
                bonds form a tree, like mother-daughter bonds in snowflake yeast
        mass  : biomass in units of a newborn cell (divides at 2)
        traits: dict of extra per-cell arrays (genetic parameters, age, ...)
    """

    def __init__(self, X, grp=None, par=None, traits=None):
        X = np.atleast_2d(np.array(X, float))
        C = len(X)
        self.X = X
        self.grp = np.arange(C) if grp is None else np.asarray(grp).copy()
        self.par = -np.ones(C, int) if par is None else np.asarray(par).copy()
        self.mass = np.ones(C) + np.random.default_rng(C).random(C)
        self.traits = {k: np.asarray(v).copy() for k, v in (traits or {}).items()}
        self._links = None
        self._next_gid = int(self.grp.max()) + 1 if C else 0

    # ---- basic properties ---------------------------------------------
    @property
    def C(self):
        return len(self.X)

    def new_gid(self, n=1):
        g = np.arange(self._next_gid, self._next_gid + n)
        self._next_gid += n
        return g

    def group_sizes(self):
        """Size of the group each cell belongs to (per cell) and per group."""
        u, inv, cnt = np.unique(self.grp, return_inverse=True, return_counts=True)
        return cnt[inv], u, cnt, inv

    def links(self):
        if self._links is None:
            idx = np.nonzero(self.par >= 0)[0]
            self._links = Links(np.stack([idx, self.par[idx]], 1), self.C)
        return self._links

    def _changed(self):
        self._links = None

    # ---- division -------------------------------------------------------
    def divide(self, who, attach, rng, kick=0.0, asym=0.05):
        """Cells `who` divide.  attach[k]=True -> daughter stays bonded to mother
        (incomplete separation, clonal multicellularity)."""
        who = np.asarray(who, int)
        if len(who) == 0:
            return np.array([], int)
        n = len(who)
        newX = self.X[who].copy()
        if kick > 0:   # partitioning noise of molecules at division
            newX = np.maximum(newX + kick * np.sqrt(newX + 1e-3) *
                              rng.standard_normal(newX.shape), 0)
        frac = np.clip(0.5 + asym * rng.standard_normal(n), 0.3, 0.7)
        dmass = self.mass[who] * (1 - frac)
        self.mass[who] *= frac
        new_idx = np.arange(self.C, self.C + n)
        g = np.where(attach, self.grp[who], self.new_gid(n))
        p = np.where(attach, who, -1)
        self.X = np.vstack([self.X, newX])
        self.grp = np.concatenate([self.grp, g])
        self.par = np.concatenate([self.par, p])
        self.mass = np.concatenate([self.mass, dmass])
        for k in self.traits:
            self.traits[k] = np.concatenate([self.traits[k], self.traits[k][who]])
        self._changed()
        return new_idx

    # ---- subtree utilities ---------------------------------------------
    def children_lists(self, members):
        ch = {int(m): [] for m in members}
        for m in members:
            p = self.par[m]
            if p >= 0:
                ch[int(p)].append(int(m))
        return ch

    def subtree(self, root, ch):
        out, stack = [], [root]
        while stack:
            v = stack.pop()
            out.append(v)
            stack.extend(ch.get(v, []))
        return out

    def cut(self, cell):
        """Break the bond between `cell` and its parent: its whole branch
        becomes a new group (a multicellular propagule)."""
        members = np.nonzero(self.grp == self.grp[cell])[0]
        ch = self.children_lists(members)
        branch = self.subtree(int(cell), ch)
        self.par[cell] = -1
        self.grp[branch] = self.new_gid()[0]
        self._changed()
        return branch

    def subtree_sizes(self, members):
        """Size of the branch hanging below each member cell (BFS order)."""
        ch = self.children_lists(members)
        roots = [int(m) for m in members if self.par[m] < 0]
        order, stack = [], list(roots)
        while stack:
            v = stack.pop()
            order.append(v)
            stack.extend(ch[v])
        sz = {v: 1 for v in order}
        for v in reversed(order):
            p = self.par[v]
            if p >= 0:
                sz[int(p)] += sz[v]
        return sz

    def fracture(self, gid, rng, frac_range=(0.2, 0.5)):
        """Packing-stress fracture of a branched cluster (snowflake yeast):
        the bond that breaks is one whose branch holds a sizeable fraction of
        the cluster, so a multicellular, clonal propagule is released."""
        members = np.nonzero(self.grp == gid)[0]
        if len(members) < 2:
            return None
        sz = self.subtree_sizes(members)
        target = len(members) * rng.uniform(*frac_range)
        cand = [v for v in sz if self.par[v] >= 0]
        best = min(cand, key=lambda v: abs(sz[v] - target) + 1e-3 * rng.random())
        return self.cut(best)

    def release_all(self, cells):
        """Dissolve bonds: every cell in `cells` becomes a free single cell."""
        cells = np.asarray(cells, int)
        hit = np.isin(self.par, cells) | np.isin(np.arange(self.C), cells)
        self.par[hit] = -1
        self.grp[cells] = self.new_gid(len(cells))
        # cells whose parent was released keep their own subgroup id: relabel
        self.relabel_components()

    def relabel_components(self):
        """Recompute group labels as connected components of the bond tree
        (cells without bonds that share a label keep it: e.g. a patch)."""
        root = np.arange(self.C)
        for _ in range(self.C + 1):          # pointer jumping
            p = self.par[root]
            if np.all(p < 0):
                break
            root = np.where(p >= 0, p, root)
        self.grp = self.grp[root]
        self._changed()

    # ---- subsetting ------------------------------------------------------
    def keep(self, mask):
        """Keep only cells in mask (bonds to removed cells are dropped)."""
        mask = np.asarray(mask, bool)
        idx = np.nonzero(mask)[0]
        remap = -np.ones(self.C, int)
        remap[idx] = np.arange(len(idx))
        par = self.par[idx]
        par = np.where(par >= 0, remap[np.maximum(par, 0)], -1)
        self.X = self.X[idx]
        self.grp = self.grp[idx]
        self.par = par
        self.mass = self.mass[idx]
        for k in self.traits:
            self.traits[k] = self.traits[k][idx]
        self._changed()

    def keep_groups(self, gids):
        self.keep(np.isin(self.grp, gids))

    def merge_groups(self, gid_list, rng):
        """Glue several groups into one aggregate (aggregative multicellularity):
        the root of each group bonds to a random cell of the growing aggregate."""
        gid_list = list(gid_list)
        base = gid_list[0]
        for g in gid_list[1:]:
            members = np.nonzero(self.grp == g)[0]
            r = members[self.par[members] < 0][0]
            target = rng.choice(np.nonzero(self.grp == base)[0])
            self.par[r] = target
            self.grp[members] = base
        self._changed()


def group_means(values, inv, ngroups):
    """Average a per-cell quantity within groups (inv from group_sizes)."""
    s = np.bincount(inv, weights=values, minlength=ngroups)
    n = np.bincount(inv, minlength=ngroups)
    return s / np.maximum(n, 1)


# --------------------------------------------------------------------------
# 4. Helpers to pick species with a given role in the network
# --------------------------------------------------------------------------


def sole_repressor_pairs(chem, s):
    """Pairs (k, j): k active, j inactive, and k is j's ONLY active neighbour.
    A noise-driven dip of k then lets j switch on and k off: an accessible,
    single-step 'mutation' k-off/j-on (paper Appendix E)."""
    s = np.asarray(s)
    pairs = []
    for j in np.nonzero(s == 0)[0]:
        act = [k for k in chem.nbrs[j] if s[k] == 1]
        if len(act) == 1:
            pairs.append((act[0], int(j)))
    return pairs


def ancestral_state(chem, rng=None, T=30.0, Lam=1.0):
    """The state reached deterministically from the all-zero initial condition
    with a tiny random perturbation (paper's default initial condition)."""
    rng = np.random.default_rng(0) if rng is None else rng
    X0 = 0.01 * rng.random((1, chem.N))
    return chem.relax(X0, T=T, Lam=Lam)[0]


def describe():
    print(__doc__)


# --------------------------------------------------------------------------
# 5. Running independent replicate experiments in parallel
# --------------------------------------------------------------------------

def run_parallel(fn, jobs, n_jobs=None):
    """Run fn(job) for each job (a dict) -- in parallel processes if n_jobs>1.
    fn must live in an importable module (not in the notebook) so that it
    can be sent to worker processes on macOS / Windows ('spawn')."""
    import os
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
    if n_jobs == 1:
        return [fn(j) for j in jobs]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    # 'spawn' is the macOS / Windows default; using it everywhere keeps runs identical
    with ProcessPoolExecutor(max_workers=min(n_jobs, len(jobs)),
                             mp_context=mp.get_context('spawn')) as ex:
        return list(ex.map(fn, jobs))
