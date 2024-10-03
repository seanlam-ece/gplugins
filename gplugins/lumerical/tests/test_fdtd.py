from pathlib import Path

import gdsfactory as gf

from gplugins.lumerical.config import DEBUG_LUMERICAL
from gplugins.lumerical.convergence_settings import LUMERICAL_FDTD_CONVERGENCE_SETTINGS
from gplugins.lumerical.fdtd import LumericalFdtdSimulation
from gplugins.lumerical.simulation_settings import SIMULATION_SETTINGS_LUMERICAL_FDTD
from gdsfactory.generic_tech.layer_stack import get_process, get_layer_stack

def test_lumerical_fdtd_simulation():

    layerstack = get_layer_stack()
    process = get_process()

    c = gf.components.straight(length=2.0)

    SIMULATION_SETTINGS_LUMERICAL_FDTD.mesh_accuracy = 1
    SIMULATION_SETTINGS_LUMERICAL_FDTD.port_field_intensity_threshold = 1e-1
    LUMERICAL_FDTD_CONVERGENCE_SETTINGS.sparam_diff = 0.1

    sim = LumericalFdtdSimulation(
        component=c,
        layerstack=layerstack,
        process=process,
        simulation_settings=SIMULATION_SETTINGS_LUMERICAL_FDTD,
        convergence_settings=LUMERICAL_FDTD_CONVERGENCE_SETTINGS,
        run_port_convergence=False,
        run_mesh_convergence=False,
        run_field_intensity_convergence=False,
        hide=False,
        dirpath=Path(__file__).resolve().parent / "test_runs",
    )

    sim.write_sparameters(overwrite=True)
