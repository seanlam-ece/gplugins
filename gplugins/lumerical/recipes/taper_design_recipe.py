import typing

import gdsfactory as gf
import numpy as np
import pandas as pd
from gdsfactory import Component
from gdsfactory.components.taper_cross_section import taper_cross_section
from gdsfactory.config import logger
from gdsfactory.pdk import LayerStack, get_layer_stack
from gdsfactory.typings import ComponentFactory, CrossSectionSpec, PathType, WidthTypes
from pydantic import BaseModel

from gplugins.design_recipe.DesignRecipe import DesignRecipe, eval_decorator
from gplugins.lumerical.config import cm, um
from gplugins.lumerical.convergence_settings import (
    LUMERICAL_EME_CONVERGENCE_SETTINGS,
    LUMERICAL_FDTD_CONVERGENCE_SETTINGS,
    ConvergenceSettingsLumericalEme,
    ConvergenceSettingsLumericalFdtd,
)
from gplugins.lumerical.eme import LumericalEmeSimulation
from gplugins.lumerical.recipes.fdtd_recipe import FdtdRecipe
from gplugins.lumerical.simulation_settings import (
    SIMULATION_SETTINGS_LUMERICAL_FDTD,
    LUMERICAL_EME_SIMULATION_SETTINGS,
    SimulationSettingsLumericalEme,
    SimulationSettingsLumericalFdtd,
)


class RoutingTaperDesignIntent(BaseModel):
    r"""
    Design intent for routing taper design recipe

    Attributes:
        narrow_waveguide_routing_loss_per_cm: Narrow waveguide routing loss (dB/cm)
        max_reflection: Maximum reflections to consider. Anything above this threshold does not determine device selection.
                        Anything below this threshold will affect device selection.
        start_length: Starting length in length sweep (um)
        stop_length: Ending length in length sweep (um)
        num_pts: Number of points to consider in length sweep
        weights: Weights for weighted decision matrix when selecting best taper

                         |       |
                         |      /|---------
                         |    /  |
                         |  /    |
        -----------------|/      |
        cross section 1  | taper | cross section 2
        -----------------|\      |
                         |  \    |
                         |    \  |
                         |      \|---------
                         |       |

    """

    narrow_waveguide_routing_loss_per_cm: float = 3  # dB / cm
    max_reflection: float = -70  # dB

    # Length Sweep
    start_length: float = 1  # um
    stop_length: float = 200  # um
    num_pts: int = 200  # um

    # Weights for decision matrix
    weights: dict[str, float] = {"transmission": 0.4,
                                 "reflection": 0.35,
                                 "length": 0.25}

    class Config:
        arbitrary_types_allowed = True


class RoutingTaperEmeDesignRecipe(DesignRecipe):
    """
    Routing taper design recipe using eigenmode expansion method (EME).

    Attributes:
        component: Optimal component geometry
        components: Components simulated (passed onto FDTD for verification)
        length_sweep: Length sweep results
        cross_section1: Left cross section
        cross_section2: Right cross section
        recipe_setup:
            cell_hash: Hash of GDSFactory component factory
            layer_stack: PDK layerstack
            simulation_setup: EME simulation setup
            convergence_setup: EME convergence setup
            design_intent: Taper design intent parameters
        recipe_results:
            length_sweeps: Length sweep data (S21, S11) vs. device geometry
            components_settings: Component settings for each device simulated in design recipe

    """

    # Results
    component: Component | None = None  # Optimal taper component
    length_sweep: pd.DataFrame | None = None  # Length sweep results
    components: list[Component] | None = None

    def __init__(
        self,
        cell: ComponentFactory = taper_cross_section,
        cross_section1: CrossSectionSpec | None = gf.cross_section.cross_section,
        cross_section2: CrossSectionSpec | None = gf.cross_section.cross_section,
        design_intent: RoutingTaperDesignIntent | None = None,
        waveguide_loss_model: pd.DataFrame | None = None,
        additional_loss: float = 0.0,
        layer_stack: LayerStack | None = None,
        simulation_setup: SimulationSettingsLumericalEme
        | None = LUMERICAL_EME_SIMULATION_SETTINGS,
        convergence_setup: ConvergenceSettingsLumericalEme
        | None = LUMERICAL_EME_CONVERGENCE_SETTINGS,
        dirpath: PathType | None = None,
    ):
        r"""
        Set up routing taper design recipe

                         |       |
                         |      /|---------
                         |    /  |
                         |  /    |
        -----------------|/      |
        cross section 1  | taper | cross section 2
        -----------------|\      |
                         |  \    |
                         |    \  |
                         |      \|---------
                         |       |

        Parameters:
            cell: Taper cell that uses cross sections to determine both ends of the taper
            cross_section1: Left cross section
            cross_section2: Right cross section
            design_intent: Taper design intent
            waveguide_loss_model: Waveguide propagation loss  (dB/cm) vs. waveguide width (um)
                                    | width | loss  |
                                    | float | float |
            additional_loss: Constant additional loss in dB (can be from substrate leakage loss, material loss, etc.)
            layer_stack: PDK layerstack
            simulation_setup: EME simulation setup
            convergence_setup: EME convergence setup
            dirpath: Directory to save files
        """
        layer_stack = layer_stack or get_layer_stack()
        super().__init__(cell=cell, layer_stack=layer_stack, dirpath=dirpath)
        self.cross_section1 = cross_section1
        self.cross_section2 = cross_section2
        # Add information to recipe setup. NOTE: This is used for hashing
        self.recipe_setup.simulation_setup = simulation_setup
        self.recipe_setup.convergence_setup = convergence_setup
        self.recipe_setup.design_intent = design_intent or RoutingTaperDesignIntent()
        if isinstance(waveguide_loss_model, pd.DataFrame) and not waveguide_loss_model.empty:
            self.recipe_setup.waveguide_loss_model = waveguide_loss_model
        if additional_loss:
            self.recipe_setup.additional_loss = additional_loss

    @eval_decorator
    def eval(self, run_convergence: bool = True):
        r"""
        Run taper design recipe.

        1. Sweep taper geometry in EME and get best geometry and length for component.
                Best component is derived from the following (in order):
                a) The dB/cm loss for the narrow waveguide routing must match the derived dB/cm loss for the taper
                b) The component must have the lowest reflections
                c) The component must be the shortest
        2. Run FDTD simulation to extract s-params for best component

        Parameters:
            run_convergence: Run convergence if True
        """
        ss = self.recipe_setup.simulation_setup
        cs = self.recipe_setup.convergence_setup
        di = self.recipe_setup.design_intent

        eme_device_length = 5 # um

        # Sweep geometry
        components = [
            self.cell(
                cross_section1=self.cross_section1,
                cross_section2=self.cross_section2,
                length=eme_device_length,  # um
                width_type=wtype,
            )
            for wtype in typing.get_args(WidthTypes)
        ]

        # Run EME length sweep simulations to extract S21 and S11 vs. length of device
        optimal_lengths = []
        transmission_coefficients = []
        reflection_coefficients = []
        simulated_components = []
        length_sweeps = []
        mode_coupling = []
        for component in components:
            try:
                sim = LumericalEmeSimulation(
                    component=component,
                    layerstack=self.recipe_setup.layer_stack,
                    simulation_settings=ss,
                    convergence_settings=cs,
                    hide=False,  # TODO: Make global variable for switching debug modes
                    run_overall_convergence=run_convergence,
                    run_mode_convergence=run_convergence,
                    dirpath=self.dirpath,
                )
                simulated_components.append(component)
            except Exception as err:
                logger.error(
                    f"{err}\n{component.name} failed to simulate. Moving onto next component"
                )
                continue

            # Get mode coupling from fundamental to higher order modes
            coupling_coefficients = sim.get_mode_coupling(input_mode=1, max_coupled_mode=ss.num_modes)
            mode_coupling.append(coupling_coefficients)

            # Extract S21 and S11 vs. length during length sweeps
            length_sweep = sim.get_length_sweep(
                start_length=di.start_length,
                stop_length=di.stop_length,
                num_pts=di.num_pts,
            )

            # Include waveguide loss model and additional loss if available in length_sweep
            propagation_loss = [0]*di.num_pts
            if hasattr(self.recipe_setup, "waveguide_loss_model"):
                wlm = self.recipe_setup.waveguide_loss_model

                pts = component.polygons[0].points
                # Iterate in coordinates from input of taper to end of taper and average the width between two points
                # Then, calculate loss for normalized length of device
                normalized_loss = 0
                for i in range(0, np.argmin(np.abs(pts[:, 0] - eme_device_length))):
                    width = np.abs(pts[i, 1] + pts[i + 1, 1])
                    length = np.abs(pts[i, 0] - pts[i + 1, 0]) * um / cm

                    loss_db_per_cm = np.interp(width, wlm.loc[:, "width"], wlm.loc[:, "loss"])
                    normalized_loss += loss_db_per_cm * length

                for i in range(0, len(length_sweep.loc[:, "length"])):
                    propagation_loss[i] += normalized_loss / eme_device_length * length_sweep.loc[i, "length"] / um

            if hasattr(self.recipe_setup, "additional_loss"):
                for i in range(0, len(propagation_loss)):
                    propagation_loss[i] += self.recipe_setup.additional_loss

            # Get transmission and reflection values
            length = length_sweep.loc[:, "length"]
            s21 = 10 * np.log10(abs(length_sweep.loc[:, "s21"]) ** 2)
            s11 = 10 * np.log10(abs(length_sweep.loc[:, "s11"]) ** 2)

            # Update length sweep with propagation loss
            s21 -= np.array(propagation_loss)
            length_sweep.loc[:, "s21"] = 10 ** (s21 / 20)

            length_sweeps.append(length_sweep)

            # Get length of taper that has lower loss than routing loss
            try:
                ind = next(
                    k
                    for k, value in enumerate(list(s21))
                    if value > -di.narrow_waveguide_routing_loss_per_cm * length[k] / cm
                )
                optimal_lengths.append(length[ind] / um)
                transmission_coefficients.append(s21[ind])
                reflection_coefficients.append(s11[ind])
            except Exception as err:
                logger.warning(f"{err}\n"
                    + f"{component.name} cannot achieve specified routing loss of "
                    + f"-{di.narrow_waveguide_routing_loss_per_cm}dB/cm. Use maximal length of {di.stop_length}um."
                )
                optimal_lengths.append(di.stop_length)
                transmission_coefficients.append(s21[len(s21)-1])
                reflection_coefficients.append(s11[len(s11)-1])

        # Log and save optimal lengths
        results = {
            f"{simulated_components[i].name} ({simulated_components[i].settings.get('width_type', 'Shape Unknown')})": f"L: {optimal_lengths[i]} | T: {transmission_coefficients[i]} | R: {reflection_coefficients[i]}"
            for i in range(0, len(simulated_components))
        }
        with open(str(self.recipe_dirpath.resolve() / "optimal_lengths.txt"), "w") as f:
            f.write(f"{results}")
        logger.info(f"{results}")
        self.components = [
            self.cell(
                cross_section1=self.cross_section1,
                cross_section2=self.cross_section2,
                length=optimal_lengths[i],  # um
                width_type=simulated_components[i].settings.get("width_type", "sine"),
            )
            for i in range(0, len(simulated_components))
        ]

        # Get best component
        # Most optimal component is one with smallest length AND least reflections
        # If not both, choose component with lower than specified reflection or least reflections
        ind1 = optimal_lengths.index(min(optimal_lengths))
        ind2 = reflection_coefficients.index(min(reflection_coefficients))

        if ind1 == ind2 or (reflection_coefficients[ind1] < di.max_reflection):
            # Select shortest component and minimal reflections. Else, shortest component with reflection below specified
            optimal_component = self.cell(
                cross_section1=self.cross_section1,
                cross_section2=self.cross_section2,
                length=optimal_lengths[ind1],  # um
                width_type=list(typing.get_args(WidthTypes))[ind1],
            )
            opt_length_sweep_data = length_sweeps[ind1]
        else:
            # Select component with minimal reflections
            optimal_component = self.cell(
                cross_section1=self.cross_section1,
                cross_section2=self.cross_section2,
                length=optimal_lengths[ind2],  # um
                width_type=list(typing.get_args(WidthTypes))[ind2],
            )
            opt_length_sweep_data = length_sweeps[ind2]

        # Save general results
        self.component = optimal_component
        self.length_sweep = opt_length_sweep_data

        # Save recipe results
        self.recipe_results.mode_coupling = mode_coupling
        self.recipe_results.length_sweeps = length_sweeps
        self.recipe_results.components_settings = [c.settings.model_copy().model_dump() for c in self.components]

        return True


class RoutingTaperDesignRecipe(DesignRecipe):
    """
    Routing taper design recipe.

    Attributes:
        best_component: Best taper component
        cross_section1: Left cross section
        cross_section2: Right cross section
        recipe_setup:
            eme_simulation_setup: EME simulation setup
            eme_convergence_setup: EME convergence setup
            fdtd_simulation_setup: FDTD simulation setup
            fdtd_convergence_setup: FDTD convergence setup
            design_intent: Taper design intent
        recipe_results:
            results: Key figures of merit for determining the best routing taper
            cell_settings: Taper cell settings that create the best routing taper
    """

    def __init__(
        self,
        cell: ComponentFactory = taper_cross_section,
        cross_section1: CrossSectionSpec | None = gf.cross_section.cross_section,
        cross_section2: CrossSectionSpec | None = gf.cross_section.cross_section,
        design_intent: RoutingTaperDesignIntent | None = None,
        waveguide_loss_model: pd.DataFrame | None = None,
        additional_loss: float = 0.0,
        eme_simulation_setup: SimulationSettingsLumericalEme = LUMERICAL_EME_SIMULATION_SETTINGS,
        eme_convergence_setup: ConvergenceSettingsLumericalEme = LUMERICAL_EME_CONVERGENCE_SETTINGS,
        fdtd_simulation_setup: SimulationSettingsLumericalFdtd = SIMULATION_SETTINGS_LUMERICAL_FDTD,
        fdtd_convergence_setup: ConvergenceSettingsLumericalFdtd = LUMERICAL_FDTD_CONVERGENCE_SETTINGS,
        layer_stack: LayerStack | None = None,
        dirpath: PathType | None = None,
    ):
        r"""
        Set up routing taper design recipe

                         |       |
                         |      /|---------
                         |    /  |
                         |  /    |
        -----------------|/      |
        cross section 1  | taper | cross section 2
        -----------------|\      |
                         |  \    |
                         |    \  |
                         |      \|---------
                         |       |

        Parameters:
            cell: Taper cell that uses cross sections to determine both ends of the taper
            cross_section1: Left cross section
            cross_section2: Right cross section
            design_intent: Taper design intent
            waveguide_loss_model: Waveguide propagation loss (dB/cm) vs. waveguide width (um)
                                    | width | loss  |
                                    | float | float |
            additional_loss: Constant additional loss in dB (can be from substrate leakage loss, material loss, etc.)
            layer_stack: PDK layerstack
            eme_simulation_setup: EME simulation setup
            eme_convergence_setup: EME convergence setup
            fdtd_simulation_setup: FDTD simulation setup
            fdtd_convergence_setup: FDTD convergence setup
            dirpath: Directory to save files
        """
        layer_stack = layer_stack or get_layer_stack()
        super().__init__(cell=cell, layer_stack=layer_stack, dirpath=dirpath)
        self.cross_section1 = cross_section1
        self.cross_section2 = cross_section2
        # Add information to recipe setup. NOTE: This is used for hashing
        self.recipe_setup.eme_simulation_setup = eme_simulation_setup
        self.recipe_setup.eme_convergence_setup = eme_convergence_setup
        self.recipe_setup.fdtd_simulation_setup = fdtd_simulation_setup
        self.recipe_setup.fdtd_convergence_setup = fdtd_convergence_setup
        self.recipe_setup.design_intent = design_intent or RoutingTaperDesignIntent()
        if isinstance(waveguide_loss_model, pd.DataFrame) and not waveguide_loss_model.empty:
            self.recipe_setup.waveguide_loss_model = waveguide_loss_model
        if additional_loss:
            self.recipe_setup.additional_loss = additional_loss


    @eval_decorator
    def eval(self, run_convergence: bool = True):
        r"""
        Run taper design recipe.

        1. Sweep taper geometry in EME and get best geometry and length for component.
        2. Run FDTD simulation to extract s-params
        3. Select best taper based on weighted decision matrix with regards to:
            a) Transmission
            b) Reflection
            c) Device Length

        Parameters:
            run_convergence: Run convergence if True
        """
        eme_recipe = RoutingTaperEmeDesignRecipe(
            cell=self.cell,
            cross_section1=self.cross_section1,
            cross_section2=self.cross_section2,
            design_intent=self.recipe_setup.design_intent,
            waveguide_loss_model=self.recipe_setup.waveguide_loss_model if hasattr(self.recipe_setup, "waveguide_loss_model") else None,
            additional_loss=self.recipe_setup.additional_loss if hasattr(self.recipe_setup, "additional_loss") else 0.0,
            layer_stack=self.recipe_setup.layer_stack,
            simulation_setup=self.recipe_setup.eme_simulation_setup,
            convergence_setup=self.recipe_setup.eme_convergence_setup,
            dirpath=self.dirpath,
        )
        eme_recipe.override_recipe = self.override_recipe
        success = eme_recipe.eval(run_convergence=run_convergence)

        fdtd_recipes = [
            FdtdRecipe(
                component=self.cell(
                    cross_section1=self.cross_section1,
                    cross_section2=self.cross_section2,
                    length=settings["length"],  # um
                    width_type=settings["width_type"],
                ),
                layer_stack=self.recipe_setup.layer_stack,
                simulation_setup=self.recipe_setup.fdtd_simulation_setup,
                convergence_setup=self.recipe_setup.fdtd_convergence_setup,
                dirpath=self.dirpath,
            )
            for settings in eme_recipe.recipe_results.components_settings
        ]
        for recipe in fdtd_recipes:
            recipe.override_recipe = self.override_recipe
            success = success and recipe.eval(run_convergence=run_convergence)

        # Select best taper based on weighted decision matrix:
        # 1. Transmission
        # 2. Reflection
        # 3. Device Length

        # Normalize weights to range between 0 to 1
        weight_names = list(self.recipe_setup.design_intent.weights.keys())
        weight_values = np.array(list(self.recipe_setup.design_intent.weights.values()))
        norm_weight_values = weight_values / sum(weight_values)
        norm_weight_values = list(norm_weight_values)

        # Get average transmission and reflection across wavelength range
        average_transmissions = []
        average_reflections = []
        for i in range(0, len(fdtd_recipes)):
            average_transmissions.append(10*np.log10(np.mean(np.abs(fdtd_recipes[i].recipe_results.sparameters.loc[:, "S21"]) ** 2)))
            average_reflections.append(10*np.log10(np.mean(np.abs(fdtd_recipes[i].recipe_results.sparameters.loc[:, "S11"]) ** 2)))

        # Get component names and optimal lengths
        taper_geometry_names = [f"{settings['width_type']}" for settings in eme_recipe.recipe_results.components_settings]
        lengths = [settings["length"] for settings in eme_recipe.recipe_results.components_settings]

        # Save results
        self.recipe_results.results = pd.DataFrame([average_transmissions, average_reflections, lengths], weight_names, taper_geometry_names)
        self.recipe_results.results.to_csv(str(self.recipe_dirpath.resolve() / "results.csv"))

        # Normalize results to range between 0 to 1 and ensure the more optimal devices have higher numbers
        # Ex. Shorter lengths are more desirable, but for decision matrix calculations, we require a positive scale
        average_transmissions = np.array(average_transmissions)
        average_reflections = -np.array(average_reflections)
        lengths = -np.array(lengths)

        norm_average_transmissions = (average_transmissions - np.min(average_transmissions)) / (np.max(average_transmissions) - np.min(average_transmissions))
        norm_average_reflections = (average_reflections - np.min(average_reflections)) / (np.max(average_reflections) - np.min(average_reflections))
        norm_lengths = (lengths - np.min(lengths)) / (np.max(lengths) - np.min(lengths))

        # Multiply by weights
        weighted_transmissions = norm_average_transmissions * norm_weight_values[0]
        weighted_reflections = norm_average_reflections * norm_weight_values[1]
        weighted_lengths = norm_lengths * norm_weight_values[2]
        weighted_score = [sum([weighted_transmissions[i], weighted_reflections[i], weighted_lengths[i]])
                          for i in range(0, len(weighted_transmissions))]

        # Save decision matrix results
        self.recipe_results.weighted_decision_matrix = pd.DataFrame([list(weighted_transmissions),
                                                 list(weighted_reflections),
                                                 list(weighted_lengths),
                                                 weighted_score
                                                 ], weight_names + ["total_score"], taper_geometry_names)
        self.recipe_results.weighted_decision_matrix.to_csv(str(self.recipe_dirpath.resolve() / "weighted_decision_matrix.csv"))

        best_component_index = weighted_score.index(max(weighted_score))
        self.best_component = taper_cross_section(**eme_recipe.recipe_results.components_settings[best_component_index])
        self.recipe_results.cell_settings = self.best_component.to_dict()

        return success

