from gplugins.lumerical.device import LumericalChargeSimulation
from gplugins.lumerical.convergence_settings import LUMERICAL_CHARGE_CONVERGENCE_SETTINGS
from gplugins.lumerical.simulation_settings import SimulationSettingsLumericalCharge
from gplugins.lumerical.config import DEBUG_LUMERICAL
from gdsfactory import logger

def test_device():
    from functools import partial
    import gdsfactory as gf
    from gdsfactory.components.straight_pin import straight_pn

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

    doped_path = gf.Path()
    doped_path.append(gf.path.arc(radius=10.0, angle=361))
    c << doped_path.extrude(cross_section=cross_section_pn)


    c = straight_pn(length=50.0)

    ### TODO: Update generic PDK with dopants in layer_stack
    from gdsfactory.generic_tech.layer_map import LAYER
    from gdsfactory.pdk import get_layer_stack
    from gdsfactory.generic_tech.layer_stack import get_process
    from gdsfactory.technology.layer_stack import LayerLevel

    layer_stack = get_layer_stack()
    process = get_process()

    for i in range(2, 10):
        process[i].layers_and = [(1,0)]

    process[2].ion = 'n'
    process[3].ion = 'n'
    process[4].ion = 'p'
    process[5].ion = 'p'
    process[6].ion = 'p'
    process[7].ion = 'n'
    process[8].ion = 'p'
    process[9].ion = 'n'

    ### Set up simulation settings
    charge_settings = SimulationSettingsLumericalCharge(x=2, y=0, yspan=8, dimension="2D X-Normal")
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
        hide=not DEBUG_LUMERICAL,
    )
    boundary_settings = {
        "anode": {
            "name": "N+",
        },
        "cathode": {
            "name": "P+",
        },
    }
    sim.set_boundary_conditions(boundary_settings)
    try:
        boundary_settings = {
            "N": {
                "name": "P",
            },
            "P": {
                "name": "N",
            },
        }
        sim.set_boundary_conditions(boundary_settings)
    except KeyError as err:
        logger.info("Correctly caught issue with swapping boundary condition names.")

