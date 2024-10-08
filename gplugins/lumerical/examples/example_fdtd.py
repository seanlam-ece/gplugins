from functools import partial

import gdsfactory as gf
from gdsfactory.components.taper_cross_section import taper_cross_section

from gplugins.lumerical.convergence_settings import LUMERICAL_FDTD_CONVERGENCE_SETTINGS
from gplugins.lumerical.fdtd import LumericalFdtdSimulation
from gplugins.lumerical.simulation_settings import SIMULATION_SETTINGS_LUMERICAL_FDTD
from gdsfactory.generic_tech.layer_stack import get_layer_stack, get_process

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

SIMULATION_SETTINGS_LUMERICAL_FDTD.port_translation = 1.0
sim = LumericalFdtdSimulation(
    component=taper,
    layerstack=layerstack,
    process=process,
    convergence_settings=LUMERICAL_FDTD_CONVERGENCE_SETTINGS,
    simulation_settings=SIMULATION_SETTINGS_LUMERICAL_FDTD,
    hide=False,
    run_port_convergence=False,
    run_mesh_convergence=False,
)

sp = sim.write_sparameters(overwrite=True)
print(sp)
