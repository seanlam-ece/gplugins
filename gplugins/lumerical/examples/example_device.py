
from functools import partial
from gplugins.lumerical.simulation_settings import SimulationSettingsLumericalCharge
from gplugins.lumerical.convergence_settings import LUMERICAL_CHARGE_CONVERGENCE_SETTINGS
from gplugins.lumerical.device import LumericalChargeSimulation
from gdsfactory.generic_tech.layer_stack import get_process
import gdsfactory as gf
from gdsfactory.pdk import get_layer_stack

# Create curved PN junction
c = gf.Component()
cross_section_pn = partial(
    gf.cross_section.pn,
    width_doping=2.425,
    width_slab=2 * 2.425,
    layer_via="VIAC",
    width_via=0.5,
    layer_metal="M1",
    width_metal=0.5,
)

p = gf.path.arc(radius=10, angle=135)
c = p.extrude(cross_section_pn)

# Get process and layerstack
process = get_process()
layer_stack = get_layer_stack()

# Set up simulation settings
charge_settings = SimulationSettingsLumericalCharge(x=10, y=10)
boundary_settings = {
    "b0": {
        "name": "anode",
        "bc mode": "steady state",
        "sweep type": "single",
        "force ohmic": True,
        "voltage": 0,
    },
    "b1": {
        "name": "cathode",
        "bc mode": "steady state",
        "sweep type": "range",
        "force ohmic": True,
        "range start": 0,
        "range stop": -10,
        "range num points": 21,
        "range backtracking": "disabled",
    },
}

sim = LumericalChargeSimulation(
    component=c,
    layerstack=layer_stack,
    process=process,
    simulation_settings=charge_settings,
    convergence_settings=LUMERICAL_CHARGE_CONVERGENCE_SETTINGS,
    boundary_settings=boundary_settings,
    hide=False,
)