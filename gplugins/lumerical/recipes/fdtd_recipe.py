from pathlib import Path

import gdsfactory as gf
from gdsfactory import Component
from gdsfactory.pdk import LayerStack, get_layer_stack

from gplugins.design_recipe.DesignRecipe import DesignRecipe, eval_decorator
from gplugins.lumerical.convergence_settings import (
    LUMERICAL_FDTD_CONVERGENCE_SETTINGS,
    ConvergenceSettingsLumericalFdtd,
)
from gplugins.lumerical.fdtd import LumericalFdtdSimulation
from gplugins.lumerical.simulation_settings import (
    SIMULATION_SETTINGS_LUMERICAL_FDTD,
    SimulationSettingsLumericalFdtd,
)


class FdtdRecipe(DesignRecipe):
    """
    FDTD recipe that extracts sparams

    Attributes:
        recipe_setup:
            simulation_setup: FDTD simulation setup
            convergence_setup: FDTD convergence setup
        dirpath: Root directory where all recipes are run
        recipe_dirpath: Recipe directory where results from recipe are stored
        recipe_results: s-parameter results.
    """
    def __init__(
        self,
        component: Component | None = None,
        layer_stack: LayerStack | None = None,
        process: tuple | None = None,
        simulation_setup: SimulationSettingsLumericalFdtd
        | None = SIMULATION_SETTINGS_LUMERICAL_FDTD,
        convergence_setup: ConvergenceSettingsLumericalFdtd
        | None = LUMERICAL_FDTD_CONVERGENCE_SETTINGS,
        dirpath: Path | None = None,
    ):
        """
        Set up FDTD recipe

        Parameters:
            component: Component
            layer_stack: PDK layer stack
            process: Process (etch, grow, implant, etc.) that affects layerstack
            simulation_setup: FDTD simulation setup
            convergence_setup: FDTD convergence setup
            dirpath: Directory to store files.
        """
        layer_stack = layer_stack or get_layer_stack()
        super().__init__(cell=component, layer_stack=layer_stack, process=process, dirpath=dirpath)
        # Add information to recipe setup. NOTE: This is used for hashing
        self.recipe_setup.simulation_setup = simulation_setup
        self.recipe_setup.convergence_setup = convergence_setup

    @eval_decorator
    def eval(self, run_convergence: bool = True) -> bool:
        """
        Run FDTD recipe to extract sparams

        1. Performs port convergence by resizing ports to ensure E-field intensities decay to specified threshold
        2. Performs mesh convergence to ensure sparams converge to certain sparam_diff
        3. Extracts sparams after updating simulation with optimal simulation params

        Parameters:
            run_convergence: Run convergence if True

        Returns:
            success: True if recipe completed properly
        """
        sim = LumericalFdtdSimulation(
            component=self.cell,
            layerstack=self.recipe_setup.layer_stack,
            process=self.recipe_setup.process,
            simulation_settings=self.recipe_setup.simulation_setup,
            convergence_settings=self.recipe_setup.convergence_setup,
            dirpath=self.dirpath,
            hide=False,  # TODO: Make global var to decide when to show sims
            run_mesh_convergence=run_convergence,
            run_port_convergence=run_convergence,
            run_field_intensity_convergence=run_convergence,
        )

        self.recipe_results.sparameters = sim.write_sparameters(
            overwrite=True, delete_fsp_files=False, plot=True
        )
        self.recipe_results.filepath_dat = sim.filepath_dat

        success = True
        return success
