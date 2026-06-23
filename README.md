# 3-Step Map Alignment Simulation

A Python implementation of the **3-Step Map Alignment Model** (Savier et al., 2017) for simulating the development of topographic neural maps between the retina, superior colliculus (SC), and primary visual cortex (V1).

The model reproduces how molecular gradients of Eph receptors and ephrin ligands guide retinal ganglion cell (RGC) axons to their correct targets during the formation of the **Retino-Collicular (RC)** and **Cortico-Collicular (CC)** maps.

The update code is more performant, able to simulate more than 50,000 connections between the source and target tissues. It is also now able to simulate the full 2D scope of topographic maps between the retina and colliculus (retino-collicular or RC map), as well as the subsequent and aligned topographic map between the primary visual cortex and the colliculus (cortico-collicular or CC map).
---

## Background

During visual system development, RGC axons must find their precise targets in the superior colliculus to create an ordered, continuous map of visual space. This is governed largely by complementary gradients of Eph receptors (on RGC axons) and ephrin ligands (in the SC). This codebase simulates that process in 2D, allowing users to:

- Model wildtype map formation
- Introduce genetic mutations (knock-ins and knock-outs) via Isl2-mediated targeting
- Visualise predicted phenotypes via simulated focal injection experiments
- Generate video animations of injection experiments sweeping across the tissue

---

## Files

```
├── mapper.py              # Core Mapper and Tissue classes; defines the simulation engine
├── sim_tools.py           # High-level functions for setting up and running simulations
├── gradients_setup.py     # Interactive script for exploring gradient configurations
├── tri_inject_sim.py      # Interactive triple focal injection experiment simulator
└── vid_tri_inj.py         # Generates MP4 animations of injection experiments
```

### `mapper.py`

The core of the 2D refinement algorithm. The Tissue class defines the gradients to be used for topographic map refinement in a given experimental condition. The Mapper class takes those gradients and generates the resultant refined map, based on an algorithm that minimizes map energy as defined Tisigankov and Koulakov.

Contains the two core classes:

- **`Tissue`** — represents a 2D neural tissue (retina, SC, or V1). Stores molecular gradient arrays for EphA, EphB, ephrinA, and ephrinB. Provides methods to construct standard gradients (`make_std_grads`), apply Isl2-mediated knock-ins (`make_isl2_ki`) and knock-outs (`make_isl2_ko`), and commit gradients to the tissue object (`set_gradients`).
- **`Mapper`** — drives the map refinement simulation. Initialises a random topographic map (`init_random_map`), builds a dataframe representation of axon-target connections (`make_map_df`), and iteratively refines it based on Eph/ephrin binding energies (`refine_map`). Also provides visualisation utilities including `df_show_grads` and `fractional_axes`.

### `sim_tools.py`

The core of the 3 Step Map Alignment Model. First the RC map is simulated (Step 1), whose connections are used to project the gradients of efnA/B from the retina into the SC (Step 2). These projected gradients are then subsequently used as the target gradients for the CC map (Step 3). This model attempts to define the mechanisms of the apparent alignment between the RC and CC maps, that persists even when grossly perturbed in Isl2-driven mutant conditions.

Wraps the core classes into convenient simulation pipelines:

- **`make_std_tissues`** — instantiates Retina, SC, and V1 tissue objects with standard wildtype gradients.
- **`run_map_sim`** — runs the full 3-step pipeline: refine the RC map, transpose retinal ephrins into the SC, then refine the CC map.
- **`sim_efnA_ki / sim_efnA_ko`** — simulate Isl2-driven ephrinA knock-in or knock-out mutations.
- **`sim_EphA_ki / sim_EphA_ko`** — simulate Isl2-driven EphA knock-in or knock-out mutations.
- **`save_grad_pics`** — saves gradient visualisation figures as PNG files.

### `gradients_setup.py`
A standalone interactive script for manually configuring gradients and exploring single-map phenotypes. Includes a cursor-following triple injection visualiser for the RC map.

### `tri_inject_sim.py`
Runs the full RC and CC simulation for a defined mutant phenotype and opens an interactive matplotlib window. Moving the cursor over either tissue panel updates the simulated injection site in real time. Holding a modifier key switches between anterograde and retrograde injection modes.

### `vid_tri_inj.py`
Loads pre-simulated mutant phenotypes from a pickle file and renders a video (MP4) of a triple injection experiment as the injection site sweeps around a circular path through the tissue. Requires FFmpeg.

---

## Example Output
Shows difference between the EphA receptor-mediated and the efnA ligand-mediated mutant phenotypes. 

### Large Isl2-Mediated EphA Knockin Mutant
- EphA-kiki-3-Cortico-ColliculuarMap-Anterograde.mp4
- EphA-kiki-3-Cortico-ColliculuarMap-Retrograde.mp4
- EphA_3ki_cc.png
- EphA_3ki_rc.png
  
### Large Isl2-Mediated ephrin-A Knockin Mutant
- efnA-kiki-0.5-Cortico-ColliculuarMap-Anterograde.mp4
- efnA-kiki-0.5-Cortico-ColliculuarMap-Retrograde.mp4
- efnA_0.5ki_cc.png
- efnA_0.5ki_rc.png

---

## Installation

**Python 3.9+** is recommended.

```bash
pip install numpy matplotlib seaborn
```

For video export (`vid_tri_inj.py`), **FFmpeg** must be installed and accessible on your system PATH:

- macOS: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`
- Windows: download from [ffmpeg.org](https://ffmpeg.org)

---

## Usage

### Running a Wildtype Simulation with Interactive Injection

```bash
python tri_inject_sim.py
```

This runs the full 3-step simulation at `Num=250` resolution and opens two interactive windows — one for the RC map and one for the CC map. Move the cursor over the source tissue to update the predicted injection phenotype in real time.

### Defining a Mutant Phenotype

Open `tri_inject_sim.py` and uncomment the relevant mutation block before running:

```python
# Isl2-EphA knock-in (5x overexpression in Isl2+ cells)
mutant_part, retina.EphA_dict = retina.make_isl2_ki('EphA Large', 5, retina.EphA_dict)

# ephrinA5 knock-out
retina.efnA_dict['efnA5'] *= 0
colliculus.efnA_dict['efnA5'] *= 0
```

Multiple mutations can be combined before calling `run_map_sim`.

### Using the Simulation API Directly

```python
from sim_tools import make_std_tissues, run_map_sim

sim_params = {
    'gamma': 100,
    'alpha': 220,
    'beta': 220,
    'R': 0.11,
    'd': 3 / 250**2,
    'Num': 250,
    'show_grads_bool': False,
    'complex_transpose': False,
}

retina, colliculus, cortex = make_std_tissues(sim_params)
rc, cc, rc_fig_grads, cc_fig_grads = run_map_sim(retina, colliculus, cortex, sim_params)
```

### Generating a Video

Edit `vid_tri_inj.py` to point `fname` at a valid pickle file containing pre-simulated mutant frames, then run:

```bash
python vid_tri_inj.py
```

Output MP4 files are saved to the working directory.

---

## Key Parameters

| Parameter | Description | Typical Value |
|---|---|---|
| `Num` | Grid resolution (Num × Num cells per tissue) | 250–350 |
| `gamma` | Weight of interstitial branching term | 100 |
| `alpha` | EphA/ephrinA interaction strength | 220 |
| `beta` | EphB/ephrinB interaction strength | 220 |
| `R` | Radius of axon competition neighbourhood | 0.11 |
| `d` | Diffusion / neighbour-coupling term | 3/Num² |
| `complex_transpose` | Use modified ephrin transposition mechanism | False |

Increasing `Num` improves spatial resolution but increases runtime roughly as O(Num²). `Num=250` is a practical default for interactive use; `Num=350` is suitable for publication-quality video output.

---

## Simulation Pipeline

```
Retina          Superior Colliculus       Visual Cortex
  │                     │                      │
  │   EphA/ephrinA       │                      │
  │   EphB/ephrinB       │   EphA/ephrinA       │
  └──── Step 1: Refine RC Map ──────────────────┘
                         │
              Step 2: Transpose retinal
              ephrins into SC coordinate space
                         │
  Cortex ──── Step 3: Refine CC Map ────► SC
```

The transposition step (Step 2) is the key biological insight of the model: retinal ephrin concentrations, carried by RGC axon terminals, are projected into the SC and subsequently guide corticotectal axon targeting.

---

## Reference

> Savier, E., et al. (2017). *A molecular mechanism for the topographic alignment of convergent neural maps.* eLife. https://doi.org/10.7554/eLife.20470
