# From a minimal physical system to multicellularity

Jupyter notebooks that extend the model of **Bunin & Rivoire, *Evolutionary features in a minimal
physical system* (arXiv:2411.03843)** to reproduce, in simulation, the evolutionary phenomena reviewed in
**Bozdag, Tong, Yunker, Herron & Ratcliff, *Overcoming key challenges in the evolutionary transition to
multicellularity* (Nature 658, 2026)**.

## The idea in one paragraph

In Bunin & Rivoire, a network of catalysed reactions with sparse, reciprocal, ultrasensitive inhibition
has a huge number of stable compositions (maximal independent sets of the inhibition graph). Noise makes
rare jumps between them, biased toward more "productive" states, and diffusion lets states spread in
space. Here each **cell is a small reactor running that chemistry**: its stable state is its heritable
composition and the noise-driven jumps are its mutations. We add only physical book-keeping —
growth, division, mother–daughter bonds, packing fracture, patches and floods, nutrient gradients —
and every multicellular phenotype is a *read-out of species that already exist in the unicellular
network* (a "separase" whose loss leaves daughters attached, an "adhesin", an "elongation" species, a
"matrix" species, a starvation-response species). No genes, templates or designed regulators are needed
for any of the three challenges in the review.

## What is in the folder

| File | What it does | Laptop run time* |
|---|---|---|
| `00_minimal_physical_system.ipynb` | Re-implements the paper's engine and checks it against Figs. 1G/S1, S7, S5, 2, 4, 5 | ~6–8 min |
| `01_group_formation.ipynb` | **Challenge 1.** Clonal (separase-loss, snowflake) vs aggregative (adhesin) groups under settling, predation and random transfer; replicate convergence; relatedness; waiting time vs mutation supply | ~5–8 min |
| `02_darwinian_threshold.ipynb` | **Challenge 2.** Biophysical scaffolding (packing-driven life cycle), group-level heritability > cell-level, MuLTEE-like selection on cluster size, ecological scaffolding with fitness decoupling, *C. flexa*-like collective inversion | ~8–12 min |
| `03_functional_integration.ipynb` | **Challenge 3.** Temporal→spatial redeployment of a starvation state as soma, environmental induction → developmental control, clonal vs chimeric development (cheaters), facultative vs obligate life cycles | ~8–12 min |
| `protoevo.py` | Shared engine: Euler–Maruyama for Eq. 1 with demographic + diffusion noise, on any graph of compartments; colony/group machinery | — |
| `nb1_model.py`, `nb2_model.py`, `nb3_model.py` | The experiment-specific models, kept in modules so replicates can run in parallel processes | — |
| `figures/` | Every figure is also saved here when you run a notebook | — |

\*On a 4-core laptop with the default settings. Each notebook has `FAST = True` (roughly half the time,
fewer replicates) and `N_JOBS` (number of parallel processes; `None` = all cores but one, `1` = serial)
at the top. The delivered notebooks already contain their outputs, so you can read them without running.

## How to run (macOS)

1. **Unzip** the folder anywhere, e.g. `~/multicell`.
2. **Create an environment** (any of these):
   ```bash
   cd ~/multicell
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
   or with conda: `conda create -n multicell python=3.11 numpy scipy matplotlib networkx jupyterlab -y && conda activate multicell`
3. **Start Jupyter from inside the folder** (the notebooks import `protoevo.py` and the `nb*_model.py`
   files from the current directory):
   ```bash
   jupyter lab        # or: jupyter notebook
   ```
4. Open the notebooks in order (00 → 03) and choose **Run → Run All Cells**. Notebook 00 is optional
   for the others but explains the engine.

To run without opening the browser:
```bash
jupyter nbconvert --to notebook --execute --inplace 01_group_formation.ipynb
```

Python ≥ 3.9 with numpy, scipy, matplotlib and networkx is all that is needed.

## Map from the review to the simulations

| Review (Bozdag et al.) | Where | Mechanism in the extended model |
|---|---|---|
| Groups evolve rapidly and convergently (Fig. 2c: 10/10 yeast populations) | 01 | Under settling selection every replicate evolves groups within a few tens of generations (most do under predation); none under random transfer |
| Single large-effect change, repurposing (ACE2 loss → incomplete separation) | 01 | One noise-driven state switch turns OFF a pre-existing "separase" species |
| Clonal vs aggregative routes (Fig. 2a) | 01 | Separase loss (clonal, kin groups) vs adhesin gain (aggregates as related as random cells) |
| Life cycle "for free" by biophysical scaffolding (Fig. 3b right) | 02-A | Branching growth + packing threshold ⇒ fracture ⇒ clonal multicellular propagules |
| Emergent multicellular heritability, higher than cellular (Zamani-Dahaj 2023, Fig. 3c left) | 02-B | Cluster size averages noisy cell shapes ⇒ H²(group) > H²(cell) |
| Selection on emergent size reshapes cells (MuLTEE) | 02-C | Settling selection on clusters drives the elongation species' catalytic supply upward |
| Ecological scaffolding (*J. sacculi*, *P. fluorescens*; Black et al. 2020) | 02-D | Patches + floods + bottlenecks make a costly matrix species spread; cell fitness falls while patch fitness rises (fitness decoupling); patch-level parent–offspring resemblance |
| Collective behaviour without communication (*C. flexa*, Fig. 3c right) | 02-E | Uncoupled two-species switches in each cell; a bistable sheet inverts all-or-none |
| Temporal-to-spatial transition (RLS1 → *regA*, Fig. 4a) | 03-A,B,E | A starvation-induced bistable state of single cells becomes position-dependent soma inside groups, cued by internal nutrient gradients |
| Environmental induction as an intermediate (*Eudorina*) | 03-B,E | Ancestral groups make soma only under external stress; evolved groups make it in rich conditions |
| Clonality supports reproductive altruism | 03-D | Clonal development evolves and keeps a germ–soma division of labour; chimeric development does not and erodes it |
| Body size and the cost of specialisation | 03-C | Interior cells are nutrient-limited, so making them soma is cheap; an interior optimum in θ |
| Obligate vs facultative multicellularity (Future directions) | 03-F | Unicellular phases counter-select the soma allele, faster for longer phases |

## Modelling choices you may want to change

* **Mutation supply** is set by the per-cell noise `omega` (cell volume Ω = 1/ω²). We use relatively large
  values so that evolution happens in tens of generations rather than hundreds; 01 shows the scaling.
* **Speed:** cells use `dt = 0.01–0.02` (the paper uses 0.01) and `N = 20–30` species (the paper uses 50).
  Notebook 00 uses N = 50 to match the paper.
* **Physical rules** (packing threshold `n0·<AR>^β`, nutrient profile in a sphere, survival function of
  soma fraction) are deliberately simple, documented in the docstrings of the `nb*_model.py` files, and
  exposed as function arguments.
* What the review lists as open problems (open-ended growth in complexity, genome expansion under small
  effective population size) is outside this model, which has a finite, predefined state space — the
  limitation Bunin & Rivoire themselves point out.
