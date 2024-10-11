from pathlib import Path

import gdsfactory as gf
from gdsfactory import logger

from gplugins.lumerical.convergence_settings import ConvergenceSettingsLumericalFdtd
from gplugins.lumerical.recipes.fdtd_recipe import FdtdRecipe
from gplugins.lumerical.simulation_settings import SimulationSettingsLumericalFdtd
from functools import partial
from gdsfactory.components.taper_cross_section import taper_cross_section

### 0. DEFINE WHERE FILES ARE SAVED
dirpath = Path(r"../recipes/recipe_runs")
dirpath.mkdir(parents=True, exist_ok=True)

### 1. DEFINE DESIGN
xs_wg = partial(
    gf.cross_section.cross_section,
    layer=(1, 0),
    width=0.5,
)

xs_wg_wide = partial(
    gf.cross_section.cross_section,
    layer=(1, 0),
    width=2.0,
)

taper = taper_cross_section(
    cross_section1=xs_wg,
    cross_section2=xs_wg_wide,
    length=5,
    width_type="parabolic",
)

### 2. DEFINE LAYER STACK
from gdsfactory.generic_tech.layer_stack import get_layer_stack, get_process
layerstack = get_layer_stack()
process = get_process()

# Remove metal and nitride layers for visualization in Lumerical
layerstack.layers.pop("nitride")
layerstack.layers.pop("via_contact")
layerstack.layers.pop("metal1")
layerstack.layers.pop("heater")
layerstack.layers.pop("via1")
layerstack.layers.pop("metal2")
layerstack.layers.pop("via2")
layerstack.layers.pop("metal3")


### 3. DEFINE SIMULATION AND CONVERGENCE SETTINGS
fdtd_simulation_setup = SimulationSettingsLumericalFdtd(
    mesh_accuracy=2, port_translation=1.0, solver_type="gpu", frequency_dependent_profile=False
)
fdtd_convergence_setup = ConvergenceSettingsLumericalFdtd(
    port_field_intensity_threshold=1e-4, sparam_diff=0.01
)

### 4. CREATE AND RUN DESIGN RECIPE
recipe = FdtdRecipe(
    component=taper,
    layer_stack=layerstack,
    process=process,
    convergence_setup=fdtd_convergence_setup,
    simulation_setup=fdtd_simulation_setup,
    dirpath=dirpath,
)

success = recipe.eval()

if success:
    logger.info("Completed FDTD recipe.")
else:
    logger.info("Incomplete run of FDTD recipe.")


