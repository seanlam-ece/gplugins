
from functools import partial
from gplugins.lumerical.simulation_settings import SimulationSettingsLumericalCharge
from gplugins.lumerical.convergence_settings import LUMERICAL_CHARGE_CONVERGENCE_SETTINGS
from gplugins.lumerical.device import LumericalChargeSimulation
import gdsfactory as gf

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

# doped_path = gf.Path()
# doped_path.append(gf.path.arc(radius=10.0, angle=361))
p = gf.path.arc(radius=10, angle=135)
c = p.extrude(cross_section_pn)
# c << doped_path.extrude(cross_section=cross_section_pn)

### TODO: Update generic PDK with dopants in layer_stack
from gdsfactory.generic_tech.layer_map import LAYER
from gdsfactory.pdk import get_layer_stack
from gdsfactory.technology.layer_stack import LayerLevel

layer_stack = get_layer_stack()
layer_stack.layers["substrate"].info["layer_type"] = "background"
layer_stack.layers["substrate"].info["background_doping_ion"] = None
layer_stack.layers["substrate"].info["background_doping_concentration"] = None
layer_stack.layers["box"].info["layer_type"] = "background"
layer_stack.layers["clad"].info["layer_type"] = "background"
layer_stack.layers["core"].sidewall_angle = 0
layer_stack.layers["slab90"].sidewall_angle = 0
layer_stack.layers["via_contact"].sidewall_angle = 0
layer_stack.layers["N"] = LayerLevel(
    layer=LAYER.N,
    thickness=0.22,
    zmin=0,
    material="si",
    mesh_order=4,
    background_doping_concentration=5e17,
    background_doping_ion="n",
    orientation="100",
    info={"layer_type": "doping"},
)
layer_stack.layers["P"] = LayerLevel(
    layer=LAYER.P,
    thickness=0.22,
    zmin=0,
    material="si",
    mesh_order=4,
    background_doping_concentration=7e17,
    background_doping_ion="p",
    orientation="100",
    info={"layer_type": "doping"},
)
layer_stack.layers["NP"] = LayerLevel(
    layer=LAYER.NP,
    thickness=0.09,
    zmin=0,
    material="si",
    mesh_order=4,
    background_doping_concentration=3e18,
    background_doping_ion="n",
    orientation="100",
    info={"layer_type": "doping"},
)
layer_stack.layers["PP"] = LayerLevel(
    layer=LAYER.PP,
    thickness=0.09,
    zmin=0,
    material="si",
    mesh_order=4,
    background_doping_concentration=2e18,
    background_doping_ion="p",
    orientation="100",
    info={"layer_type": "doping"},
)
layer_stack.layers["NPP"] = LayerLevel(
    layer=LAYER.NPP,
    thickness=0.09,
    zmin=0,
    material="si",
    mesh_order=4,
    background_doping_concentration=1e19,
    background_doping_ion="n",
    orientation="100",
    info={"layer_type": "doping"},
)
layer_stack.layers["PPP"] = LayerLevel(
    layer=LAYER.PPP,
    thickness=0.09,
    zmin=0,
    material="si",
    mesh_order=4,
    background_doping_concentration=1e19,
    background_doping_ion="p",
    orientation="100",
    info={"layer_type": "doping"},
)

# c.show()

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
    simulation_settings=charge_settings,
    convergence_settings=LUMERICAL_CHARGE_CONVERGENCE_SETTINGS,
    boundary_settings=boundary_settings,
    hide=False,
)