from gplugins.lumerical.simulation_settings import LUMERICAL_MODE_SIMULATION_SETTINGS
from gplugins.lumerical.convergence_settings import LUMERICAL_MODE_CONVERGENCE_SETTINGS
from gplugins.lumerical.mode import LumericalModeSimulation
from gplugins.lumerical.config import DEBUG_LUMERICAL
from pathlib import Path

def test_mode():
    import gdsfactory as gf
    c = gf.components.straight()

    from gdsfactory.generic_tech.layer_stack import get_process
    from gdsfactory.pdk import get_layer_stack

    layer_stack = get_layer_stack()
    process = get_process()

    LUMERICAL_MODE_SIMULATION_SETTINGS.x = 2
    LUMERICAL_MODE_SIMULATION_SETTINGS.y = 0
    LUMERICAL_MODE_SIMULATION_SETTINGS.z = 0.11
    LUMERICAL_MODE_SIMULATION_SETTINGS.xspan = 2
    LUMERICAL_MODE_SIMULATION_SETTINGS.zspan = 1
    LUMERICAL_MODE_SIMULATION_SETTINGS.injection_axis = "2D X normal"
    LUMERICAL_MODE_SIMULATION_SETTINGS.mesh_cells_per_wavl = 30
    LUMERICAL_MODE_SIMULATION_SETTINGS.num_modes = 5
    LUMERICAL_MODE_SIMULATION_SETTINGS.wavl_pts = 4

    LUMERICAL_MODE_CONVERGENCE_SETTINGS.ng_diff = 0.1
    LUMERICAL_MODE_CONVERGENCE_SETTINGS.neff_diff = 0.1
    LUMERICAL_MODE_CONVERGENCE_SETTINGS.pol_diff = 0.01

    sim = LumericalModeSimulation(component=c,
                                  layerstack=layer_stack,
                                  process=process,
                                  simulation_settings=LUMERICAL_MODE_SIMULATION_SETTINGS,
                                  convergence_settings=LUMERICAL_MODE_CONVERGENCE_SETTINGS,
                                  run_mesh_convergence=True,
                                  run_port_convergence=True,
                                  override_convergence=False,
                                  dirpath=Path(__file__).resolve().parent / "test_runs",
                                  hide=not DEBUG_LUMERICAL)
    sim.plot_index()
    sim.plot_neff_vs_wavelength()
    sim.plot_ng_vs_wavelength()
    sim.plot_dispersion_vs_wavelength()