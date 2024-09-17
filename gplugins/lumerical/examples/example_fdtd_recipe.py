from pathlib import Path

import gdsfactory as gf
from gdsfactory import logger

from gplugins.lumerical.convergence_settings import ConvergenceSettingsLumericalFdtd
from gplugins.lumerical.recipes.fdtd_recipe import FdtdRecipe
from gplugins.lumerical.simulation_settings import SimulationSettingsLumericalFdtd
from functools import partial
from gdsfactory.components.taper_cross_section import taper_cross_section

### 0. DEFINE WHERE FILES ARE SAVED
dirpath = Path("../recipes/recipe_runs")
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
from gdsfactory.technology.layer_stack import LayerLevel, LayerStack, LogicalLayer

layerstack_lumerical = LayerStack(
    layers={
        "clad": LayerLevel(
            layer=LogicalLayer(layer=(600, 0)),
            thickness=3.0,
            zmin=0.0,
            material="sio2",
            sidewall_angle=0.0,
            mesh_order=9,
            info={"layer_type": "background"},
        ),
        "box": LayerLevel(
            layer=LogicalLayer(layer=(600, 0)),
            thickness=3.0,
            zmin=-3.0,
            material="sio2",
            sidewall_angle=0.0,
            mesh_order=9,
            info={"layer_type": "background"},
        ),
        "core": LayerLevel(
            layer=LogicalLayer(layer=(1, 0)),
            thickness=0.22,
            zmin=0.0,
            material="si",
            sidewall_angle=2.0,
            width_to_z=0.5,
            mesh_order=2,
            info={"active": True,
                  "layer_type": "grow"},
        ),
    }
)

### 3. DEFINE SIMULATION AND CONVERGENCE SETTINGS
fdtd_simulation_setup = SimulationSettingsLumericalFdtd(
    mesh_accuracy=2, port_translation=1.0
)
fdtd_convergence_setup = ConvergenceSettingsLumericalFdtd(
    port_field_intensity_threshold=1e-6, sparam_diff=0.01
)

### 4. CREATE AND RUN DESIGN RECIPE
recipe = FdtdRecipe(
    component=taper,
    layer_stack=layerstack_lumerical,
    convergence_setup=fdtd_convergence_setup,
    simulation_setup=fdtd_simulation_setup,
)

success = recipe.eval()

if success:
    logger.info("Completed FDTD recipe.")
else:
    logger.info("Incomplete run of FDTD recipe.")


