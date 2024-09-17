from gplugins.design_recipe.DesignRecipe import DesignRecipe
from pydantic import BaseModel
from gdsfactory.components.straight import straight
from gdsfactory.typings import ComponentFactory, PathType
from gdsfactory.technology.layer_stack import LayerStack
from gdsfactory.pdk import get_layer_stack
from gplugins.lumerical.simulation_settings import (
    SimulationSettingsLumericalMode,
    LUMERICAL_MODE_SIMULATION_SETTINGS,
)
from gplugins.lumerical.convergence_settings import (
    ConvergenceSettingsLumericalMode,
    LUMERICAL_MODE_CONVERGENCE_SETTINGS,
)
from gplugins.design_recipe.DesignRecipe import eval_decorator
from gplugins.lumerical.mode import LumericalModeSimulation
import pandas as pd
import numpy as np
from gplugins.lumerical.config import um, cm

class WaveguideSweepDesignIntent(BaseModel):
    r"""
    Design intent for sweeping waveguide geometry and characterizing optical characteristics

    Attributes:
        param_name: Waveguide cell parameter name
        param_vals: Parameter values to sweep the waveguide geometry
    """
    param_name: str = ""
    param_vals: list = []

    class Config:
        arbitrary_types_allowed = True

class WaveguideSweepRecipe(DesignRecipe):
    def __init__(
        self,
        cell: ComponentFactory = straight,
        design_intent: WaveguideSweepDesignIntent | None = None,
        layer_stack: LayerStack | None = None,
        simulation_setup: SimulationSettingsLumericalMode
        | None = LUMERICAL_MODE_SIMULATION_SETTINGS,
        convergence_setup: ConvergenceSettingsLumericalMode
        | None = LUMERICAL_MODE_CONVERGENCE_SETTINGS,
        dirpath: PathType | None = None,
    ):
        r"""
        Set up waveguide geometry sweep to characterize optical characteristics

        Parameters:
            cell: Waveguide cell
            design_intent: Waveguide sweep design intent
            layer_stack: PDK layerstack
            simulation_setup: MODE simulation setup
            convergence_setup: MODE convergence setup
            dirpath: Directory to save files
        """
        layer_stack = layer_stack or get_layer_stack()
        super().__init__(cell=cell, layer_stack=layer_stack, dirpath=dirpath)
        # Add information to recipe setup. NOTE: This is used for hashing
        self.recipe_setup.simulation_setup = simulation_setup
        self.recipe_setup.convergence_setup = convergence_setup
        self.recipe_setup.design_intent = design_intent or WaveguideSweepDesignIntent()

    @eval_decorator
    def eval(self, run_convergence: bool = True):
        r"""
        Run taper design recipe.

        1. Sweep waveguide geometry in MODE and extract relevant waveguide parameters

        Parameters:
            run_convergence: Run convergence if True
        """
        di = self.recipe_setup.design_intent
        ss = self.recipe_setup.simulation_setup

        # Set up sweep
        params = [{di.param_name: val} for val in di.param_vals]

        neff = []
        ng = []
        te_polarization = []
        for param in params:
            # Update simulation setup based on component
            c = self.cell(**param)
            simulation_setup = self.recipe_setup.simulation_setup.copy()
            simulation_setup.x = c.x
            simulation_setup.y = c.y

            sim = LumericalModeSimulation(
                component=c,
                layerstack=self.recipe_setup.layer_stack,
                simulation_settings=simulation_setup,
                convergence_settings=self.recipe_setup.convergence_setup,
                dirpath=self.dirpath,
                run_port_convergence=run_convergence,
                run_mesh_convergence=run_convergence,
                hide=False, # TODO: Add global to show or hide sims
            )

            s = sim.session

            # Get modes
            s.run()
            s.mesh()
            s.findmodes()
            s.selectmode(ss.target_mode)

            # Get optical characteristics
            neff.append(s.getresult(f"FDE::data::mode{ss.target_mode}", "neff")[0][0])
            ng.append(s.getresult(f"FDE::data::mode{ss.target_mode}", "ng")[0][0])
            te_polarization.append(s.getresult(f"FDE::data::mode{ss.target_mode}", "TE polarization fraction"))

        self.recipe_results.waveguide_sweep_characteristics = pd.DataFrame({di.param_name: di.param_vals,
                                                                            "neff": neff,
                                                                            "ng": ng,
                                                                            "TE_polarization": te_polarization})
        return True


def grillot_strip_waveguide_loss_model(neff_vs_width: pd.DataFrame,
                                       wavelength: float = 1.55,
                                       Lc: float = 0.05,
                                       sigma: float = 0.002,
                                       n_core: float = 3.474,
                                       n_clad: float = 1.444,
                                       ) -> pd.DataFrame:
    """
    Get waveguide loss vs. width using the model from F. Grillot

    Reference:
        F. Grillot et al., “Influence of Waveguide Geometry on Scattering Loss Effects in Submicron Strip Silicon-on-Insulator Waveguides,”
        IET Optoelectronics 2, no. 1 (February 1, 2008): 1–5, https://digital-library.theiet.org/content/journals/10.1049/iet-opt_20070001.

    Parameters:
        neff_vs_width: Effective index vs. waveguide width (um)
                    | neff  | width |
                    | float | float |
        wavelength: Wavelength (um)
        Lc: Correlation length (um)
        sigma: Standard deviation in waveguide geometry (um)
        n_core: Refractive index of core
        n_clad: Refractive index of cladding

    Returns:
        Loss in dB/cm vs. width (um)
        | loss  | width |
        | float | float |
    """

    neff_r = np.real(neff_vs_width.loc[:, "neff"])
    widths = neff_vs_width.loc[:, "width"] * um / cm
    wavelength_cm = wavelength * um / cm
    Lc = Lc * um / cm
    sigma = sigma * um / cm

    loss_dB_per_cm = []
    for i in range(0, len(widths)):
        neff = neff_r[i]
        d = widths[i] / 2
        k0 = 2 * np.pi / wavelength_cm

        # Calculate coefficients
        U = k0 * d * np.sqrt(n_core ** 2 - neff ** 2)
        V = k0 * d * np.sqrt(n_core ** 2 - n_clad ** 2)
        W = k0 * d * np.sqrt(neff ** 2 - n_clad ** 2)
        g = (U ** 2 * V ** 2) / (1 + W)

        # Define f(x)
        delta = (n_core ** 2 - n_clad ** 2) / ( 2 * n_core ** 2)
        gamma = n_clad * V / (n_core * W * np.sqrt(delta))
        x = W * Lc / d

        f = (x * np.sqrt(1 - x ** 2 + np.sqrt((1 + x ** 2) ** 2 + (2 * x ** 2 * gamma ** 2)))) / (np.sqrt((1 + x ** 2) ** 2 + (2*x**2*gamma**2)))
        loss_dB_per_cm.append(4.34 * ((sigma ** 2) / (2 * np.pi * k0 * np.sqrt(2) * (d ** 4) * n_core)) * g * f)

    return pd.DataFrame({"width": neff_vs_width.loc[:, "width"],
                         "loss": loss_dB_per_cm})