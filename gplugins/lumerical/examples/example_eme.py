import gdsfactory as gf
from gplugins.lumerical.eme import LumericalEmeSimulation
from functools import partial
from gdsfactory.components.taper_cross_section import taper_cross_section
from gdsfactory.generic_tech.layer_stack import get_process, get_layer_stack

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
    width_type="linear",
)

layerstack = get_layer_stack()
process = get_process()

# KNOWN BUG: Lumerical Layer Builder sometimes cannot handle large sidewall angles
layerstack.layers["core"].sidewall_angle = 2.0

sim = LumericalEmeSimulation(
    taper,
    layerstack=layerstack,
    process=process,
    run_mesh_convergence=False,
    run_cell_convergence=False,
    run_overall_convergence=False,
    run_mode_convergence=False,
    hide=False,
)

data = sim.get_length_sweep()
print(data)