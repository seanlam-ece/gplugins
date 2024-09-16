from pathlib import Path
from gplugins.lumerical.recipes.waveguide_recipe import (
    WaveguideSweepRecipe,
    WaveguideSweepDesignIntent,
    grillot_strip_waveguide_loss_model,
)
from gplugins.lumerical.simulation_settings import SimulationSettingsLumericalMode
from gplugins.lumerical.convergence_settings import ConvergenceSettingsLumericalMode
from gdsfactory.config import logger
from gdsfactory.components.straight import straight
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


### 0. DEFINE WHERE FILES ARE SAVED
dirpath = Path("../recipes/recipe_runs")
dirpath.mkdir(parents=True, exist_ok=True)

### 1. DEFINE DESIGN INTENT
design_intent = WaveguideSweepDesignIntent(
    param_name="width",
    param_vals=list(np.linspace(0.5, 3.0, 26))
)

### 2. DEFINE LAYER STACK
from gdsfactory.technology.layer_stack import LayerLevel, LayerStack

layerstack_lumerical = LayerStack(
    layers={
        "box": LayerLevel(
            layer=(99999, 0),
            thickness=3.0,
            zmin=-3.0,
            material="sio2",
            sidewall_angle=0.0,
            mesh_order=9,
            layer_type="background",
        ),
        "core": LayerLevel(
            layer=(1, 0),
            thickness=0.22,
            zmin=0.0,
            material="si",
            sidewall_angle=2.0,
            width_to_z=0.5,
            mesh_order=2,
            layer_type="grow",
            info={"active": True},
        ),
    }
)

### 3. DEFINE SIMULATION AND CONVERGENCE SETTINGS
convergence_setup = ConvergenceSettingsLumericalMode(neff_diff=5e-3,
                                                     ng_diff=5e-3)
simulation_setup = SimulationSettingsLumericalMode(injection_axis="2D X normal",
                                                   mesh_cells_per_wavl=80,
                                                   num_modes=10,
                                                   target_mode=1,
                                                   )


wg_recipe = WaveguideSweepRecipe(cell=straight,
                                    design_intent=design_intent,
                                    simulation_setup=simulation_setup,
                                    convergence_setup=convergence_setup,
                                    layer_stack=layerstack_lumerical,
                                    dirpath=dirpath)
wg_recipe.override_recipe = False
success = wg_recipe.eval()
if success:
    logger.info("Completed waveguide recipe.")
else:
    logger.info("Incomplete run of waveguide recipe.")

# Get waveguide loss model
loss_model = grillot_strip_waveguide_loss_model(neff_vs_width=wg_recipe.recipe_results.waveguide_sweep_characteristics,
                                                wavelength=simulation_setup.wavl,
                                                )

plt.figure()
plt.plot(loss_model.loc[:, "width"], loss_model.loc[:, "loss"], marker="*")
plt.xlabel("Width (um)")
plt.ylabel("Loss (dB/cm)")
plt.title("Waveguide Propagation Loss")
plt.grid("on")
plt.show()