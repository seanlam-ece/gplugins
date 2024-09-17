from gdsfactory.components.coupler_ring import coupler_ring
from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.components.straight import straight
import gdsfactory as gf
from functools import partial
from gdsfactory.cross_section import (
    strip,
    Section,
    pn,
    rib,
)
from gdsfactory.typings import CrossSectionFactory
from gdsfactory.component import Component
from collections.abc import Callable
from gdsfactory import logger
from pathlib import Path
from gdsfactory.technology.layer_stack import LayerLevel, LayerStack
from gdsfactory.generic_tech.layer_map import LAYER

from gplugins.lumerical.recipes.microring_modulator_recipe import (
    PNJunctionDesignIntent,
    PNMicroringModulatorRecipe,
    PNJunctionRecipe,
)
from gplugins.lumerical.recipes.fdtd_recipe import FdtdRecipe
from gplugins.lumerical.simulation_settings import (
    SimulationSettingsLumericalCharge,
    SimulationSettingsLumericalMode,
    SIMULATION_SETTINGS_LUMERICAL_FDTD,
    LUMERICAL_INTERCONNECT_SIMULATION_SETTINGS,
)
from gplugins.lumerical.convergence_settings import (
    ConvergenceSettingsLumericalCharge,
    ConvergenceSettingsLumericalMode,
    LUMERICAL_FDTD_CONVERGENCE_SETTINGS,
)
from scipy.constants import speed_of_light
from gplugins.lumerical.config import um
import pandas as pd

@gf.cell
def ring_double_pn_2seg(
    add_gap: float = 0.6,
    thru_gap: float = 0.6,
    radius: float = 15.0,
    length_coupling: float = 0.0,
    length_pn: float = 25.0,
    waveguide_cross_section: CrossSectionFactory = rib,
    pn_cross_section: CrossSectionFactory = pn,
) -> Component:
    """Double bus ring modulator with 2 segments of PN junction phaseshifters.

    NOTE: N++, N+ overlaps with N doping areas, and P++, P+ overlaps with P
    doping areas.


                    Top View of Double Bus Ring Modulator
              DROP  ────────────────────────────────────  ADD
              PORT  ────────────────────────────────────  PORT
                                                 ▲
                              length_coupling    │ add_gap
                                  <------>       ▼
                                  ────────       ▲
                                /   ────   \     │
                               /  /      \  \    │  radius
                              /  /        \  \   │
                             /  /          \  \  ▼
                            │  │            │  │ ▲
                            │  │            │  │ │
                            │  │            │  │ │ length_pn
                            │  │            │  │ │
                            │  │            │  │ ▼
                             \  \          /  /  ▲
                              \  \        /  /   │
                               \  \      /  /    │ radius
                                \   ────   /     │
                                  ────────       ▼
                                  <------>       ▲
                              length_coupling    │ thru_gap
                                                 ▼
              IN    ────────────────────────────────────  THRU
             PORT   ────────────────────────────────────  PORT

    Args:
        add_gap: gap to add waveguide.
        thru_gap: gap to drop waveguide.
        radius: bend radius for coupler
        length_coupling: length of coupling region
        length_pn: length of PN junction phaseshifters
        waveguide_cross_section: waveguide cross section
        pn_cross_section: PN junction cross section

    """
    c = gf.Component()

    # Create couplers
    thru_coupler = coupler_ring(gap=thru_gap,
                            radius=radius,
                            length_x=length_coupling,
                            bend=bend_circular,
                            length_extension=0,
                            cross_section=waveguide_cross_section,
                            )
    add_coupler = coupler_ring(gap=add_gap,
                                radius=radius,
                                length_x=length_coupling,
                                bend=bend_circular,
                                length_extension=0,
                                cross_section=waveguide_cross_section,
                                )

    ref_thru_coupler = c.add_ref(thru_coupler)
    ref_add_coupler = c.add_ref(add_coupler)

    # Shift device such that 0,0 is at port o1 (bottom left)
    ref_thru_coupler.movex(radius)
    ref_add_coupler.movex(-radius)

    # Create PN phase shifters
    pn_junction = partial(straight,
                          cross_section=pn_cross_section,
                          length=length_pn)

    phase_shifter1 = pn_junction()
    ps1 = phase_shifter1.rotate(90)
    ref_ps1 = c.add_ref(ps1)
    ref_ps1.x = ref_thru_coupler.ports["o2"].center[0]
    ref_ps1.ymin = ref_thru_coupler.ports["o2"].center[1]

    ref_ps2 = c.add_ref(ps1)
    ref_ps2.rotate(180)
    ref_ps2.x = ref_thru_coupler.ports["o3"].center[0]
    ref_ps2.ymin = ref_thru_coupler.ports["o3"].center[1]

    # Place add coupler above the PN phase shifters
    ref_add_coupler.rotate(180)
    ref_add_coupler.ymin = ref_ps1.ymax

    # Add ports to component
    c.add_port("o1", port=ref_thru_coupler["o1"])
    c.add_port("o2", port=ref_thru_coupler["o4"])
    c.add_port("o3", port=ref_add_coupler["o1"])
    c.add_port("o4", port=ref_add_coupler["o4"])
    c.auto_rename_ports()

    return c

def get_waveguide_cross_section(width_ridge: float = 0.45,
                                width_slab: float = 5.45,
                                radius: float = 15,
                                ) -> Callable:
    """

                                Waveguide Cross Section
                                   width_ridge
                         |<------------------------->|
                         ┌───────────────────────────┐
                         │                           │
┌────────────────────────┘                           └─────────────────────────┐
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
|<---------------------------------------------------------------------------->|
                                   width_slab

    :param width_ridge:
    :param width_slab:
    :param radius:
    :return:
    """
    return partial(
        strip,
        width=width_ridge,
        sections=(Section(width=width_slab, layer="SLAB90", name="slab90"),),
        radius=radius,
    )

def get_pn_cross_section(width_ridge: float = 0.45,
                         width_slab: float = 5.45,
                         width_contact: float = 1.0,
                         width_doping: float = 8.0,
                         offset_low_doping: float = -0.065,
                         gap_medium_doping: float = 0.595,
                         gap_high_doping: float = 0.65,
                         waveguide_layer: tuple = (1, 0),
                         metal_layer: tuple = (40, 0),
                    ) -> Callable:
    """
                                PN Phaseshifter Cross Section
                                       width_ridge
                             |<------------------------->|
                             |         offset_low_doping |
                             |             <------->     |
                             |             |       |     |
                             |            wg    junction |
     width_contact           |           center  center  |             width_contact
    |<---------->|           |             |       |     |            |<---------->|
    ┌────────────┐           ┌─────────────|───────|─────┐            ┌────────────┐
    │    |       │           │             |       |     │            │      |     │
    │    |       └───────────┘             |       |     └────────────┘      |     │
    │    | P++  |  P+  |         P         |       |     N     |  N+  |  N++ |     │
    └────|──────|──────|───────────────────|───────────────────|──────|──────|─────┘
    |    |      |      |<----------------->|<----------------->|      |      |     |
    |    |      |      | gap_medium_doping | gap_medium_doping |      |      |     |
    |    |      |      |                   |                   |      |      |     |
    |    |      |<------------------------>|<------------------------>|      |     |
    |    |      |     gap_high_doping      |      gap_high_doping     |      |     |
    |    |<----------------------------------------------------------------->|     |
    |                                  width_doping                                |
    |<---------------------------------------------------------------------------->|
                                       width_slab



    """
    return partial(pn, width=width_ridge,
                      offset_low_doping=offset_low_doping,
                      gap_medium_doping=gap_medium_doping,
                      gap_high_doping=gap_high_doping,
                      width_doping=width_doping,
                      width_slab=width_slab,
                      sections=(Section(width=width_contact,
                                        offset=(width_slab - width_contact) / 2,
                                        layer=waveguide_layer),
                                Section(width=width_contact,
                                        offset=-(width_slab - width_contact) / 2,
                                        layer=waveguide_layer),
                                Section(width=width_contact,
                                        offset=(width_slab - width_contact) / 2,
                                        layer=metal_layer),
                                Section(width=width_contact,
                                        offset=-(width_slab - width_contact) / 2,
                                        layer=metal_layer),
                                ))

### 0. DEFINE WHERE FILES ARE SAVED
dirpath = Path("../recipes/recipe_runs")
dirpath.mkdir(parents=True, exist_ok=True)

### 1. DEFINE DESIGN INTENT
design_intent = pn_design_intent=PNJunctionDesignIntent(voltage_start=-0.8,
                                                                   voltage_stop=3,
                                                                   voltage_pts=39)


width_slab = 5.45
rings = {
    "ring1": {
            "component_settings": {
                "length_pn": 25.0
            },
            "cross_section_settings": {
                "width_ridge": 0.44,
                "width_slab": width_slab,
            }
        },
    "ring2": {
        "component_settings": {
            "length_pn": 25.05
        },
        "cross_section_settings": {
            "width_ridge": 0.448,
            "width_slab": width_slab,
        }
    },
    "ring3": {
        "component_settings": {
            "length_pn": 25.1
        },
        "cross_section_settings": {
            "width_ridge": 0.444,
            "width_slab": width_slab,
        }
    },
    "ring4": {
        "component_settings": {
            "length_pn": 25.15
        },
        "cross_section_settings": {
            "width_ridge": 0.444,
            "width_slab": width_slab,
        }
    },
}

ring_components = [ring_double_pn_2seg(pn_cross_section=get_pn_cross_section(**ring_settings["cross_section_settings"]),
                                       waveguide_cross_section=get_waveguide_cross_section(width_slab=ring_settings["cross_section_settings"]["width_slab"]),
                                       **ring_settings["component_settings"]) for ring_settings in rings.values()]
pn_components = [c.named_references["rotate_1"].parent for c in ring_components]

### 2. DEFINE LAYER STACK
core_thickness = 0.15  # um
slab_thickness = 0.05  # um

# Dopant concentrations
dopant_concentration_sets = {
    "amf": [5e17, 3e18, 1e19, 7e17, 2e18, 1e19],
    "amf_low": [5e16, 1e18, 1e19, 7e16, 1e18, 1e19],
    "amf_midlow": [5e17, 1e18, 1e19, 7e17, 1e18, 1e19],
    "sweep1": [1e17, 1e18, 1e19, 1e17, 1e18, 1e19],
    "sweep2": [1e16, 1e17, 1e18, 1e16, 1e17, 1e18],
    "sweep3": [1e18, 1e19, 1e20, 1e18, 1e19, 1e20],
    "sweep4": [1e19, 1e20, 1e21, 1e19, 1e20, 1e21], # NOTE: This requires changing target_mode to 1
    "1sweep1": [1e18, 1e19, 1e20, 1e18, 1e19, 1e20],
    "1sweep2": [2e18, 1e19, 1e20, 2e18, 1e19, 1e20],
    "1sweep3": [4e18, 1e19, 1e20, 4e18, 1e19, 1e20],
    "1sweep4": [8e17, 1e19, 1e20, 8e17, 1e19, 1e20],
    "1sweep5": [6e17, 1e19, 1e20, 6e17, 1e19, 1e20],
    "2sweep1": [1e18, 1e19, 1e20, 1e18, 1e19, 1e20],
    "2sweep2": [1e18, 2e19, 1e20, 1e18, 2e19, 1e20],
    "2sweep3": [1e18, 4e19, 1e20, 1e18, 4e19, 1e20],
    "2sweep4": [1e18, 8e19, 1e20, 1e18, 8e19, 1e20],
    "change_straggle": [1e18, 1e19, 1e20, 1e18, 1e19, 1e20],
    "carrier_fine_tune1": [1e18, 1e19, 1e20, 1e18, 1e19, 1e20],
    "carrier_fine_tune2": [1.2e18, 1e19, 1e20, 1.2e18, 1e19, 1e20],
    "carrier_fine_tune3": [1.4e18, 1e19, 1e20, 1.4e18, 1e19, 1e20],
    "carrier_fine_tune4": [1.6e18, 1e19, 1e20, 1.6e18, 1e19, 1e20],
    "carrier_fine_tune5": [1.2e18, 1e19, 1e20, 2e18, 1e19, 1e20],
}
sets = ["sweep2", "sweep1", "sweep3", "sweep4"]

for set in sets:

    n = dopant_concentration_sets[set][0]

    np = dopant_concentration_sets[set][1]
    npp = dopant_concentration_sets[set][2]

    p = dopant_concentration_sets[set][3]
    pp = dopant_concentration_sets[set][4]
    ppp = dopant_concentration_sets[set][5]

    layerstack_lumerical = LayerStack(
        layers={
            "clad": LayerLevel(
                layer=(99999, 0),
                thickness=3.0,
                zmin=0.0,
                material="sio2",
                sidewall_angle=0.0,
                mesh_order=9,
                layer_type="background",
            ),
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
                thickness=core_thickness,
                zmin=0.0,
                material="si",
                sidewall_angle=2.0,
                width_to_z=0.5,
                mesh_order=2,
                layer_type="grow",
                info={"active": True},
            ),
            "slab90": LayerLevel(
                layer=(3, 0),
                thickness=slab_thickness,
                zmin=0.0,
                material="si",
                sidewall_angle=2.0,
                width_to_z=0.5,
                mesh_order=2,
                layer_type="grow",
                info={"active": True},
            ),
            "N": LayerLevel(
                layer=LAYER.N,
                thickness=core_thickness,
                zmin=core_thickness,
                material="si",
                mesh_order=4,
                background_doping_concentration=n,
                background_doping_ion="n",
                orientation="100",
                layer_type="doping",
            ),
            "NP": LayerLevel(
                layer=LAYER.NP,
                thickness=core_thickness,
                zmin=core_thickness,
                material="si",
                mesh_order=4,
                background_doping_concentration=np,
                background_doping_ion="n",
                orientation="100",
                layer_type="doping",
            ),
            "NPP": LayerLevel(
                layer=LAYER.NPP,
                thickness=core_thickness,
                zmin=core_thickness,
                material="si",
                mesh_order=4,
                background_doping_concentration=npp,
                background_doping_ion="n",
                orientation="100",
                layer_type="doping",
            ),
            "P": LayerLevel(
                layer=LAYER.P,
                thickness=core_thickness,
                zmin=core_thickness,
                material="si",
                mesh_order=4,
                background_doping_concentration=p,
                background_doping_ion="p",
                orientation="100",
                layer_type="doping",
            ),
            "PP": LayerLevel(
                layer=LAYER.PP,
                thickness=core_thickness,
                zmin=core_thickness,
                material="si",
                mesh_order=4,
                background_doping_concentration=pp,
                background_doping_ion="p",
                orientation="100",
                layer_type="doping",
            ),
            "PPP": LayerLevel(
                layer=LAYER.PPP,
                thickness=core_thickness,
                zmin=core_thickness,
                material="si",
                mesh_order=4,
                background_doping_concentration=ppp,
                background_doping_ion="p",
                orientation="100",
                layer_type="doping",
            ),
            "via": LayerLevel(
                layer=LAYER.VIAC,
                thickness=1.0,
                zmin=core_thickness,
                material="Aluminum",
                mesh_order=4,
                orientation="100",
                layer_type="grow",
            ),
        }
    )

    layerstack_no_doping = LayerStack(
        layers={
            "clad": LayerLevel(
                layer=(99999, 0),
                thickness=3.0,
                zmin=0.0,
                material="sio2",
                sidewall_angle=0.0,
                mesh_order=9,
                layer_type="background",
            ),
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
                thickness=core_thickness,
                zmin=0.0,
                material="si",
                sidewall_angle=2.0,
                width_to_z=0.5,
                mesh_order=2,
                layer_type="grow",
                info={"active": True},
            ),
            "slab90": LayerLevel(
                layer=(3, 0),
                thickness=slab_thickness,
                zmin=0.0,
                material="si",
                sidewall_angle=2.0,
                width_to_z=0.5,
                mesh_order=2,
                layer_type="grow",
                info={"active": True},
            )}
    )

    # 3. DEFINE SIMULATION AND CONVERGENCE SETTINGS
    charge_settings = [SimulationSettingsLumericalCharge(x=c.x,
                                                        y=c.y,
                                                        dimension="2D Y-Normal",
                                                        xspan=width_slab + 1.0) for c in pn_components]
    charge_convergence_settings = ConvergenceSettingsLumericalCharge(
        global_iteration_limit=500,
        gradient_mixing="fast")

    mode_settings = [SimulationSettingsLumericalMode(injection_axis="2D Y normal",
                                                    wavl_pts=11,
                                                    x=settings.x,
                                                    y=settings.y,
                                                    z=settings.z,
                                                    xspan=settings.xspan,
                                                    yspan=settings.yspan,
                                                    zspan=settings.zspan,
                                                    target_mode=1 if (set == "sweep4") else 3,
                                                    ) for settings in charge_settings]
    mode_convergence_settings = ConvergenceSettingsLumericalMode()

    # 4. CREATE AND RUN DESIGN RECIPES
    mrm_recipes = []
    for i in range(0, len(ring_components)):
        mrm_recipes.append(PNMicroringModulatorRecipe(component=ring_components[i],
                                                layer_stack=layerstack_lumerical,
                                                pn_design_intent=design_intent,
                                                mode_simulation_setup=mode_settings[i],
                                                mode_convergence_setup=mode_convergence_settings,
                                                charge_simulation_setup=charge_settings[i],
                                                charge_convergence_setup=charge_convergence_settings,
                                                fdtd_simulation_setup=SIMULATION_SETTINGS_LUMERICAL_FDTD,
                                                fdtd_convergence_setup=LUMERICAL_FDTD_CONVERGENCE_SETTINGS,
                                                interconnect_simulation_setup=LUMERICAL_INTERCONNECT_SIMULATION_SETTINGS,
                                                dirpath=dirpath,
                                                dependencies=[PNJunctionRecipe(component=ring_components[i].named_references["rotate_1"].parent,
                                                                 layer_stack=layerstack_lumerical,
                                                                 design_intent=pn_design_intent,
                                                                 mode_simulation_setup=mode_settings[i],
                                                                 mode_convergence_setup=mode_convergence_settings,
                                                                 charge_simulation_setup=charge_settings[i],
                                                                 charge_convergence_setup=charge_convergence_settings,
                                                                 dirpath=dirpath),
                                            FdtdRecipe(component=ring_components[i].named_references["coupler_ring_1"].parent,
                                                       layer_stack=layerstack_no_doping,
                                                       simulation_setup=SIMULATION_SETTINGS_LUMERICAL_FDTD,
                                                       convergence_setup=LUMERICAL_FDTD_CONVERGENCE_SETTINGS,
                                                       dirpath=dirpath)]
                                                )
                           )
        if i == 0:
            mrm_recipes[0].dependencies.constituent_recipes[0].override_recipe = False
            mrm_recipes[0].dependencies.constituent_recipes[0].dependencies.constituent_recipes[0].override_recipe = False
    # Run recipes
    for mrm_recipe in mrm_recipes:
        mrm_recipe.override_recipe = False
        mrm_recipe.eval()


# Build ring circuit
import lumapi
s = lumapi.INTERCONNECT(hide=False)

compound_names = []
for i in range(0, len(mrm_recipes)):
    recipe = mrm_recipes[i]
    dirpath = recipe.recipe_dirpath

    # Set custom folder to look at models here
    s.loadcustom(str(dirpath))

    # Add models to create ring
    wg1 = s.addelement(f"::custom::{recipe.recipe_results.waveguide_model_settings.settings['name'].lower()}")
    wg2 = s.addelement(f"::custom::{recipe.recipe_results.waveguide_model_settings.settings['name'].lower()}")
    ps1 = s.addelement(f"::custom::{recipe.recipe_results.phaseshifter_model_settings.settings['name'].lower()}")
    ps2 = s.addelement(f"::custom::{recipe.recipe_results.phaseshifter_model_settings.settings['name'].lower()}")
    cp1 = s.addelement(f"::custom::{recipe.recipe_results.coupler_model_settings.settings['name'].lower()}")
    cp2 = s.addelement(f"::custom::{recipe.recipe_results.coupler_model_settings.settings['name'].lower()}")
    wg1_name = wg1.name
    wg2_name = wg2.name
    ps1_name = ps1.name
    ps2_name = ps2.name
    cp1_name = cp1.name
    cp2_name = cp2.name

    s.connect(cp1_name, "o3", wg1_name, "port 1")
    s.connect(wg1_name, "port 2", ps1_name, "port 1")
    s.connect(ps1_name, "port 2", cp2_name, "o2")
    s.connect(cp2_name, "o3", wg2_name, "port 1")
    s.connect(wg2_name, "port 2", ps2_name, "port 1")
    s.connect(ps2_name, "port 2", cp1_name, "o2")

    # Create compound element from elements
    s.select(wg1_name)
    s.shiftselect(wg2_name)
    s.shiftselect(ps1_name)
    s.shiftselect(ps2_name)
    s.shiftselect(cp1_name)
    s.shiftselect(cp2_name)

    compound_name = s.createcompound()
    port1 = s.addport(compound_name, "port1", "Bidirectional", "Optical Signal",
                      "Left", 0.33)
    port2 = s.addport(compound_name, "port2", "Bidirectional", "Optical Signal",
                      "Left", 0.66)

    port3 = s.addport(compound_name, "port3", "Bidirectional", "Optical Signal",
                      "Right", 0.33)
    port4 = s.addport(compound_name, "port4", "Bidirectional", "Optical Signal",
                      "Right", 0.66)

    e1 = s.addport(compound_name, "e1", "Bidirectional", "Electrical Signal",
                      "Top", 0.5)

    s.groupscope(compound_name)
    s.connect("RELAY_1", "port", cp1_name, "o1")
    s.connect("RELAY_2", "port", cp2_name, "o4")
    s.connect("RELAY_3", "port", cp1_name, "o4")
    s.connect("RELAY_4", "port", cp2_name, "o1")
    s.connect("RELAY_5", "port", ps1_name, "modulation")
    s.connect("RELAY_5", "port", ps2_name, "modulation")
    s.groupscope("::Root Element")

    # Rename compound element
    s.select(compound_name)
    s.set("name", recipe.cell.name)
    s.set("description", str(recipe.cell.settings))
    s.set("x position", 200*i)
    s.unselectall()
    compound_names.append(recipe.cell.name)

# Connect rings in a weightbank
for i in range(0, len(compound_names) - 1):
    s.connect(compound_names[i], "port3", compound_names[i + 1], "port1")
    s.connect(compound_names[i], "port4", compound_names[i + 1], "port2")

# Add electrical drive to modulators
dc_elements = []
for i in range(0, len(compound_names)):
    dc_elements.append(s.addelement("DC Source"))
    s.set("x position", 200*i)
    s.set("y position", -200)
    s.connect(compound_names[i], "e1", dc_elements[-1].name, "output")

osa = s.addelement("Optical Network Analyzer", properties={
            "number of input ports": 3,
            "input parameter": "start and stop",
            "start frequency": speed_of_light / (LUMERICAL_INTERCONNECT_SIMULATION_SETTINGS.wavl_start * um),
            "stop frequency": speed_of_light / (LUMERICAL_INTERCONNECT_SIMULATION_SETTINGS.wavl_end * um),
            "number of points": LUMERICAL_INTERCONNECT_SIMULATION_SETTINGS.wavl_pts,
        })
s.set("y position", -500)
s.connect(osa.name, "output", compound_names[0], "port1")
s.connect(osa.name, "input 1", compound_names[0], "port2")
s.connect(osa.name, "input 2", compound_names[-1], "port3")
s.connect(osa.name, "input 3", compound_names[-1], "port4")

# Run simulation and extract optical response
s.run()
s.select(osa.name)

# Get THRU port response
data = s.getresult(osa.name, "input 1/mode 1/gain")
thru_port_data = pd.DataFrame(
    {"wavelength": data["wavelength"][:, 0] / um,
     "gain": data["mode 1 gain (dB)"]})

# Get DROP port response
data = s.getresult(osa.name, "input 2/mode 1/gain")
drop_port_data = pd.DataFrame(
    {"wavelength": data["wavelength"][:, 0] / um,
     "gain": data["mode 1 gain (dB)"]})

# Get ADD port response
data = s.getresult(osa.name, "input 3/mode 1/gain")
add_port_data = pd.DataFrame(
    {"wavelength": data["wavelength"][:, 0] / um,
     "gain": data["mode 1 gain (dB)"]})

# Set up packages
import numpy as np
import pandas as pd
import matplotlib as mpl
from gplugins.lumerical.config import marker_list
from gplugins.lumerical.interconnect import get_free_spectral_range, \
    get_resonances

mpl.use("Qt5Agg")
import matplotlib.pyplot as plt

# Selects which ring to analyze from simulations
ring_index = 2

# Get experimental results
exp_dirpath = Path("./experimental_results")

optical_spectrum_weight_bank = pd.read_csv(
    str(exp_dirpath.resolve() / "optical_spectrum_weight_bank.csv"),
    index_col=0)
optical_spectrum_resonance_vs_voltage = pd.read_csv(
    str(exp_dirpath.resolve() / "optical_spectrum_resonance_vs_voltage.csv"),
    index_col=0)
resonance_shift = pd.read_csv(
    str(exp_dirpath.resolve() / "resonance_shift.csv"), index_col=0)

fsr_data = get_free_spectral_range(
    wavelength=list(optical_spectrum_weight_bank.loc[:, "wavelength"]),
    power=list(10 * np.log10(optical_spectrum_weight_bank.loc[:, "drop"])),
    prominence=5)

start_resonance = 3
exp_fsr = (fsr_data.loc[start_resonance + 4, "peak_wavelength"] - fsr_data.loc[
    start_resonance, "peak_wavelength"]) * 1e-3
sim_fsr = mrm_recipes[ring_index].recipe_results.free_spectral_range

# Fit calibration spectrum to experimental data
coefficients = np.polyfit(optical_spectrum_weight_bank.loc[:, "wavelength"],
                          10 * np.log10(
                              optical_spectrum_weight_bank.loc[:, "thru"]),
                          2)

p = np.poly1d(coefficients)
power_fit = p(optical_spectrum_weight_bank.loc[:, "wavelength"])

# Plot weight bank spectrum
fig1 = plt.figure()
plt.plot(optical_spectrum_weight_bank.loc[:, "wavelength"],
         10 * np.log10(optical_spectrum_weight_bank.loc[:, "thru"]) - power_fit,
         marker=marker_list[0],
         label="Experimental - THRU")
plt.plot(optical_spectrum_weight_bank.loc[:, "wavelength"],
         10 * np.log10(optical_spectrum_weight_bank.loc[:, "drop"]) - power_fit,
         marker=marker_list[1],
         label="Experimental - DROP")
# plt.plot(optical_spectrum_weight_bank.loc[:,"wavelength"],
#          power_fit,
#          marker=marker_list[4],
#          label="Experimental - Calibration Fit")
plt.plot(thru_port_data.loc[:, "wavelength"] * 1e3,
         thru_port_data.loc[:, "gain"],
         marker=marker_list[2],
         label="Simulation - THRU")
plt.plot(drop_port_data.loc[:, "wavelength"] * 1e3,
         drop_port_data.loc[:, "gain"],
         marker=marker_list[3],
         label="Simulation - DROP")
plt.xlim([min(optical_spectrum_weight_bank.loc[:, "wavelength"]),
          max(optical_spectrum_weight_bank.loc[:, "wavelength"])])
plt.xlabel("Wavelength (nm)")
plt.ylabel("Power (dBm)")
plt.title("1x4 MRM Spectra")
plt.grid("on")
plt.legend()
plt.savefig(str(exp_dirpath / "sim_vs_exp_1x4_mrm_spectra.png"))

# Plot optical spectrum vs voltage applied
fig2 = plt.figure()
i = 0
for name in optical_spectrum_resonance_vs_voltage.columns:
    if "wavelength" in name:
        voltage = name.split("_")[-1]
        plt.plot(optical_spectrum_resonance_vs_voltage.loc[:, name],
                 optical_spectrum_resonance_vs_voltage.loc[:, voltage],
                 label=f"Experimental - {voltage}",
                 marker=marker_list[i])
        i += 1

plt.xlabel("Wavelength (nm)")
plt.ylabel("Power (dB)")
plt.title("Peak 3")
plt.grid("on")
plt.legend()

# Plot wavelength and power shifts
fig3, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(resonance_shift.loc[:, "voltage"],
         resonance_shift.loc[:, "wavelength_shift"],
         marker=marker_list[0],
         label="Experimental")
ax1.set_xlabel("Voltage (V)")
ax1.set_ylabel("Wavelength Shift (nm)")
ax1.set_title("Wavelength Shift")
ax1.legend()
ax1.grid(True)

ax2.plot(resonance_shift.loc[:, "voltage"],
         resonance_shift.loc[:, "power_shift"],
         marker=marker_list[0],
         label="Experimental")
ax2.set_xlabel("Voltage (V)")
ax2.set_ylabel("Power Shift (dB)")
ax2.set_title("Power Shift")
ax2.grid(True)

# Get resonance shift from simulation
wavl_range = [1.572, 1.577]
sub_spectrums = []
sim_wavelength_resonances = []
sim_power_resonances = []
for i in range(0, len(mrm_recipes[ring_index].recipe_results.spectrum_vs_voltage.loc[:,
                      "spectrum"])):
    spectrum = mrm_recipes[ring_index].recipe_results.spectrum_vs_voltage.loc[i, "spectrum"]
    sub_spectrums.append(spectrum[(spectrum["wavelength"] >= wavl_range[0]) & (
                spectrum["wavelength"] <= wavl_range[1])])
    resonances = get_resonances(
        wavelength=list(sub_spectrums[-1].loc[:, "wavelength"]),
        power=list(sub_spectrums[-1].loc[:, "drop"]))
    sim_wavelength_resonances.append(resonances.loc[0, "resonant_wavelength"])
    sim_power_resonances.append(resonances.loc[0, "resonant_power"])

sim_wavelength_resonances = np.array(sim_wavelength_resonances)
sim_power_resonances = np.array(sim_power_resonances)
norm_sim_wavelength_resonances = sim_wavelength_resonances - \
                                 sim_wavelength_resonances[0]
norm_sim_power_resonances = sim_power_resonances - sim_power_resonances[0]

ax1.plot(mrm_recipes[ring_index].recipe_results.spectrum_vs_voltage.loc[:, "voltage"],
         norm_sim_wavelength_resonances * 1e3,
         marker=marker_list[1],
         label="Simulation",
         )
ax1.legend()
ax2.plot(mrm_recipes[ring_index].recipe_results.spectrum_vs_voltage.loc[:, "voltage"],
         norm_sim_power_resonances,
         marker=marker_list[1],
         label="Simulation",
         )
ax2.legend()
fig3.savefig(str(exp_dirpath / "sim_vs_exp_resonance_shift.png"))

plt.show()
logger.info("Done")