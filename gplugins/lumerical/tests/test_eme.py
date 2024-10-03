from pathlib import Path

import gdsfactory as gf
from gdsfactory.generic_tech.layer_stack import get_layer_stack, get_process

from gplugins.lumerical.config import DEBUG_LUMERICAL
from gplugins.lumerical.eme import LumericalEmeSimulation
from gplugins.lumerical.simulation_settings import LUMERICAL_EME_SIMULATION_SETTINGS
from gplugins.lumerical.convergence_settings import LUMERICAL_EME_CONVERGENCE_SETTINGS


def test_lumerical_eme_simulation_setup():
    layer_stack = get_layer_stack()
    process = get_process()

    LUMERICAL_EME_SIMULATION_SETTINGS.num_modes = 10
    LUMERICAL_EME_SIMULATION_SETTINGS.mesh_cells_per_wavelength = 30

    LUMERICAL_EME_CONVERGENCE_SETTINGS.sparam_diff = 0.1

    c = gf.components.straight()

    sim = LumericalEmeSimulation(
        component=c,
        layerstack=layer_stack,
        process=process,
        simulation_settings=LUMERICAL_EME_SIMULATION_SETTINGS,
        convergence_settings=LUMERICAL_EME_CONVERGENCE_SETTINGS,
        run_mode_convergence=True,
        run_overall_convergence=True,
        hide=not DEBUG_LUMERICAL,
        dirpath=Path(__file__).resolve().parent / "test_runs",
    )

    sim.plot_length_sweep()
