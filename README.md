# Updated 2D 3 Step Map Alignment Model
Code implementing The Updated 3 Step Map Alignment Algorithm of topographic map refinement, based on mathematical modeling done by Tsigankov and Koulakov. 

The update code is more performant, able to simulate more than 50,000 connections between the source and target tissues. It is also now able to simulate the full 2D scope of topographic maps between the retina and colliculus (retino-collicular or RC map), as well as the subsequent and aligned topographic map between the primary visual cortex and the colliculus (cortico-collicular or CC map). 

## Core Utilities
### mapper.py
The core of the 2D refinement algorithm. The Tissue class defines the gradients to be used for topographic map refinement in a given experimental condition. The Mapper class takes those gradients and generates the resultant refined map, based on an algorithm that minimizes map energy as defined Tisigankov and Koulakov. 

### sim_tools.py
The core of the 3 Step Map Alignment Model. First the RC map is simulated (Step 1), whose connections are used to project the gradients of efnA/B from the retina into the SC (Step 2). These projected gradients are then subsequently used as the target gradients for the CC map (Step 3). This model attempts to define the mechanisms of the apparent alignment between the RC and CC maps, that persists even when grossly perturbed in Isl2-driven mutant conditions. 

## Drivers of Computational Simulation
### gradients_setup.py 
Drives the mapper.py and sim_tools.py code to generate a specific mutant condition. 

### vid_tri_inject.py & tri_inject_sim.py
Uses the simulated topographic maps to generate figures representing tripple injection experiments. Injections can be performed in the source or target tissue, representing anterograde and retrograde focal injection experiments, respectively. 

## Example Outputs 
Shows difference between the EphA-Mediated and the efn-A Mediated Mutant Phenotypes

### Large Isl2-Mediated EphA Knockin Mutant 
EphA-kiki-3-Cortico-ColliculuarMap-Anterograde.mp4
EphA-kiki-3-Cortico-ColliculuarMap-Retrograde.mp4
EphA_3ki_cc.png
EphA_3ki_rc.png

### Large Isl2-Mediated ephrin-A Knockin Mutant 
efnA-kiki-0.5-Cortico-ColliculuarMap-Anterograde.mp4
efnA-kiki-0.5-Cortico-ColliculuarMap-Retrograde.mp4
efnA_0.5ki_cc.png
efnA_0.5ki_rc.png