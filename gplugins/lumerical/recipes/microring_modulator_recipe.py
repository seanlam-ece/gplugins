

from pathlib import Path

import numpy as np
import pandas as pd

from gdsfactory.component import Component

from gdsfactory import logger
from pydantic import BaseModel

from gplugins.design_recipe.DesignRecipe import DesignRecipe, eval_decorator
from gdsfactory.pdk import LayerStack, get_layer_stack
from gplugins.lumerical.device import LumericalChargeSimulation
from gplugins.lumerical.simulation_settings import (
    SimulationSettingsLumericalCharge,
    LUMERICAL_CHARGE_SIMULATION_SETTINGS,
    SimulationSettingsLumericalMode,
    LUMERICAL_MODE_SIMULATION_SETTINGS,
    SimulationSettingsLumericalFdtd,
    SIMULATION_SETTINGS_LUMERICAL_FDTD,
    SimulationSettingsLumericalInterconnect,
    LUMERICAL_INTERCONNECT_SIMULATION_SETTINGS,
)
from gplugins.lumerical.convergence_settings import (
    ConvergenceSettingsLumericalCharge,
    LUMERICAL_CHARGE_CONVERGENCE_SETTINGS,
    ConvergenceSettingsLumericalMode,
    LUMERICAL_MODE_CONVERGENCE_SETTINGS,
    ConvergenceSettingsLumericalFdtd,
    LUMERICAL_FDTD_CONVERGENCE_SETTINGS,
)
from gplugins.lumerical.mode import LumericalModeSimulation
from gplugins.lumerical.interconnect import (
    get_resonances,
    get_free_spectral_range,
    create_compact_model,
)
from gplugins.lumerical.config import um
from gplugins.lumerical.recipes.fdtd_recipe import FdtdRecipe
from gplugins.lumerical.compact_models import (
    WAVEGUIDE_COMPACT_MODEL,
    PHASESHIFTER_COMPACT_MODEL,
    SPARAM_COMPACT_MODEL,
)
from scipy.constants import speed_of_light
from gdsfactory.technology.processes import ImplantPhysical, ImplantGaussian, DopingConstant
from gdsfactory.pdk import get_layer_tuple


class PNJunctionDesignIntent(BaseModel):
    r"""
    Design intent for PN junction

    Attributes:
        contact1_name: Name of first contact connected to PN junction
        contact2_name: Name of second contact connected to PN junction
        voltage_start: Start voltage in sweep
        voltage_stop: Stop voltage in sweep
        voltage_pts: Number of voltage points in sweep
    """
    contact1_name: str = "anode"
    contact2_name: str = "cathode"
    voltage_start: float = -1
    voltage_stop: float = 3
    voltage_pts: int = 21

    class Config:
        arbitrary_types_allowed = True


class PNJunctionChargeRecipe(DesignRecipe):
    """
    PN junction CHARGE recipe.

    Attributes:
        recipe_setup:
            simulation_setup: CHARGE simulation setup
            convergence_setup: CHARGE convergence setup
            design_intent: PN junction design intent
        recipe_results:
            charge_profile_path: Path to charge profile file (.mat)
            ac_voltage: Small signal AC voltage used to characterize AC response
            frequency: Small signal frequency used to characterize AC response
            electrical_characteristics: Electrical characteristics of junction,
                including voltage bias vs. impedance, capacitance, resistance,
                bandwidth.
    """
    def __init__(
        self,
        component: Component | None = None,
        layer_stack: LayerStack | None = None,
        process: tuple | None = None,
        design_intent: PNJunctionDesignIntent | None = None,
        simulation_setup: SimulationSettingsLumericalCharge
        | None = LUMERICAL_CHARGE_SIMULATION_SETTINGS,
        convergence_setup: ConvergenceSettingsLumericalCharge
        | None = LUMERICAL_CHARGE_CONVERGENCE_SETTINGS,
        dirpath: Path | None = None,
    ):
        layer_stack = layer_stack or get_layer_stack()
        super().__init__(cell=component, layer_stack=layer_stack, process=process,
                         dirpath=dirpath)
        # Add information to recipe setup. NOTE: This is used for hashing
        self.recipe_setup.simulation_setup = simulation_setup
        self.recipe_setup.convergence_setup = convergence_setup
        self.recipe_setup.design_intent = design_intent

    @eval_decorator
    def eval(self, run_convergence: bool = True):
        r"""
        Run PN junction recipe

        1. Sweep PN junction voltage and extract charge profile
        2. Sweep PN junction voltage and extract AC response and electrical
            characteristics

        Parameters:
            run_convergence: Run convergence if True
        """
        # Set up simulation to extract charge profile vs. voltage
        boundary_settings = {
            "b0": {
                "name": self.recipe_setup.design_intent.contact2_name,
                "bc mode": "steady state",
                "sweep type": "range",
                "force ohmic": True,
                "range start": self.recipe_setup.design_intent.voltage_start,
                "range stop": self.recipe_setup.design_intent.voltage_stop,
                "range num points": self.recipe_setup.design_intent.voltage_pts,
                "range backtracking": "enabled",
            },
            "b1": {
                "name": self.recipe_setup.design_intent.contact1_name,
                "bc mode": "steady state",
                "sweep type": "single",
                "force ohmic": True,
                "voltage": 0,
            },
        }
        sim = LumericalChargeSimulation(component=self.cell,
                                        layerstack=self.recipe_setup.layer_stack,
                                        process=self.recipe_setup.process,
                                        simulation_settings=self.recipe_setup.simulation_setup,
                                        convergence_settings=self.recipe_setup.convergence_setup,
                                        boundary_settings=boundary_settings,
                                        dirpath=self.dirpath,
                                        hide=False,
                                        )
        s = sim.session
        s.run()
        self.recipe_results.charge_profile_path = sim.simulation_dirpath / "charge.mat"

        # Set up simulation to extract RC electrical characteristics
        s.switchtolayout()
        s.setnamed("CHARGE::monitor", "save data", 0)
        s.setnamed("CHARGE", "solver mode", "ssac")
        s.setnamed("CHARGE", "solver type", "newton")


        boundary_settings = {
            self.recipe_setup.design_intent.contact2_name: {
                "apply ac small signal": "all",
            },
        }
        sim.set_boundary_conditions(boundary_settings=boundary_settings)
        s.save()
        s.run()

        # Get electrical results
        results = s.getresult("CHARGE", f"ac_{self.recipe_setup.design_intent.contact2_name}")
        vac = s.getnamed("CHARGE", "perturbation amplitude")
        if self.recipe_setup.simulation_setup.dimension == "2D X-Normal":
            iac = results["dI"][:,0,0][:,0]
        elif self.recipe_setup.simulation_setup.dimension == "2D Y-Normal":
            iac = results["dI"][0,:,0][:,0]
        else:
            iac = results["dI"][0, 0, :][:, 0]
        vbias = results[f"V_{self.recipe_setup.design_intent.contact2_name}"][:,0]
        f = results["f"][0][0]

        Z = vac / iac
        Y = 1 / Z

        C = np.abs(np.imag(Y) / (2 * np.pi * f)) # F/cm
        R = np.abs(np.real(Z)) # ohm-cm

        BW = 1/(2 * np.pi * R * C)

        self.recipe_results.ac_voltage = vac
        self.recipe_results.frequency = f
        self.recipe_results.electrical_characteristics = pd.DataFrame({"vbias": list(vbias),
                                                                       "impedance": list(Z),
                                                                       "capacitance_F_per_cm": list(C),
                                                                       "resistance_ohm_cm": list(R),
                                                                       "bandwidth_GHz": list(BW)})
        return True


class PNJunctionRecipe(DesignRecipe):
    """
    PN junction recipe that includes optical (MODE) and electrical (CHARGE)
    analysis.

    Attributes:
        recipe_setup:
            mode_simulation_setup: MODE simulation setup
            mode_convergence_setup: MODE convergence setup
            charge_simulation_setup: CHARGE simulation setup
            charge_convergence_setup: CHARGE convergence setup
            design_intent: PN junction design intent
        recipe_results:
            charge_profile_path: Path to charge profile file (.mat)
            ac_voltage: Small signal AC voltage used to characterize AC response
            frequency: Small signal frequency used to characterize AC response
            electrical_characteristics: Electrical characteristics of junction,
                including voltage bias vs. impedance, capacitance, resistance,
                bandwidth.
            waveguide_profile_path: Path to MODE waveguide profile (waveguide.ldf)
            neff_vs_voltage: Effective index vs. voltage
            phase_loss_vs_voltage: Phase and loss per cm vs. voltage

    """
    def __init__(
        self,
        component: Component | None = None,
        layer_stack: LayerStack | None = None,
        process: tuple | None = None,
        design_intent: PNJunctionDesignIntent | None = None,
        mode_simulation_setup: SimulationSettingsLumericalMode | None = LUMERICAL_MODE_SIMULATION_SETTINGS,
        mode_convergence_setup: ConvergenceSettingsLumericalMode | None = LUMERICAL_MODE_CONVERGENCE_SETTINGS,
        charge_simulation_setup: SimulationSettingsLumericalCharge | None = LUMERICAL_CHARGE_SIMULATION_SETTINGS,
        charge_convergence_setup: ConvergenceSettingsLumericalCharge | None = LUMERICAL_CHARGE_CONVERGENCE_SETTINGS,
        dependencies: list[DesignRecipe] | None = None,
        dirpath: Path | None = None,
    ):
        layer_stack = layer_stack or get_layer_stack()
        dependencies = dependencies or [PNJunctionChargeRecipe(component=component,
                                               layer_stack=layer_stack,
                                               process=process,
                                               design_intent=design_intent,
                                               simulation_setup=charge_simulation_setup,
                                               convergence_setup=charge_convergence_setup,
                                               dirpath=dirpath)]
        super().__init__(cell=component, layer_stack=layer_stack, process=process,
                         dirpath=dirpath, dependencies=dependencies)
        # Add information to recipe setup. NOTE: This is used for hashing
        self.recipe_setup.mode_simulation_setup = mode_simulation_setup
        self.recipe_setup.mode_convergence_setup = mode_convergence_setup
        self.recipe_setup.charge_simulation_setup = charge_simulation_setup
        self.recipe_setup.charge_convergence_setup = charge_convergence_setup
        self.recipe_setup.design_intent = design_intent

    @eval_decorator
    def eval(self, run_convergence: bool = True):
        r"""
        Run PN junction recipe

        1. Sweep PN junction voltage and extract charge profile

        Parameters:
            run_convergence: Run convergence if True
        """
        # Issue: Metals affect MODE's ability to effectively calculate waveguide modes.
        # Solution: Use a different layerstack where metal layer(s) are removed
        layerstack_lumerical_mode = self.recipe_setup.layer_stack.copy()
        layer_name = self.recipe_setup.layer_stack.get_layer_to_layername()[
            self.recipe_setup.charge_simulation_setup.metal_layer]
        layerstack_lumerical_mode.layers.pop(layer_name)

        mode_sim = LumericalModeSimulation(component=self.cell,
                                           layerstack=layerstack_lumerical_mode,
                                           process=self.recipe_setup.process,
                                           simulation_settings=self.recipe_setup.mode_simulation_setup,
                                           convergence_settings=self.recipe_setup.mode_convergence_setup,
                                           dirpath=self.dirpath,
                                           hide=False)

        pn_recipe = self.dependencies.constituent_recipes[0]
        self.recipe_results.charge_profile_path = pn_recipe.recipe_results.charge_profile_path
        self.recipe_results.ac_voltage = pn_recipe.recipe_results.ac_voltage
        self.recipe_results.frequency = pn_recipe.recipe_results.frequency
        self.recipe_results.electrical_characteristics = pn_recipe.recipe_results.electrical_characteristics

        # Set up MODE sim to import charge profile
        s = mode_sim.session
        s.addgridattribute("np density")
        s.set("x", 0)
        s.set("y", 0)
        s.set("z", 0)
        s.importdataset(str(pn_recipe.recipe_results.charge_profile_path.resolve()))

        # Get layers that are doped
        for p in self.recipe_setup.process:
            if type(p) is ImplantPhysical or type(p) is ImplantGaussian or type(p) is DopingConstant:
                # Get material layers that require doping
                affected_layers = p.layers_or if p.layers_or else []
                affected_layers += p.layers_and if p.layers_and else []

                for affected_layer in affected_layers:
                    s.select("layer group")
                    layer_name = layerstack_lumerical_mode.get_layer_to_layername()[get_layer_tuple(affected_layer)]
                    base_material = s.getlayer(layer_name, "pattern material")
                    patterned = True
                    if not base_material:
                        base_material = s.getlayer(layer_name, "background material")
                        patterned = False

                    # Create material with relation between free carriers and index
                    new_material_name = f"{layer_name}_doped"
                    material_props = {"Name": new_material_name,
                                      "np density model": "Soref and Bennett",
                                      "Coefficients": "Nedeljkovic, Soref & Mashanovich, 2011",
                                      "Base Material": base_material}

                    mname = s.addmaterial("Index perturbation")
                    s.setmaterial(mname, material_props)

                    # Set layer material with free carriers
                    s.select("layer group")
                    if patterned:
                        s.setlayer(layer_name, "pattern material", new_material_name)
                    else:
                        s.setlayer(layer_name, "background material", new_material_name)

        # If convergence setup has changed or convergence results are not available
        if (not mode_sim.convergence_is_fresh() or not mode_sim.convergence_results.available()) \
                and run_convergence:
            # Run convergence sweeps to ensure simulation accuracy
            mode_sim.convergence_results.mesh_convergence_data = mode_sim.update_mesh_convergence(verbose=True,
                                             plot=True)
            # Save convergence results as to skip convergence sweeps in future runs
            mode_sim.save_convergence_results()
            logger.info("Saved convergence results.")

        # Add voltage sweep
        s.addsweep()
        s.setsweep("sweep", "number of points", self.recipe_setup.design_intent.voltage_pts)
        para = {"Name": "voltage",
                "Parameter": f"::model::np density::V_{self.recipe_setup.design_intent.contact2_name}_index",
                "Type": "Number",
                "Start": 1,
                "Stop": self.recipe_setup.design_intent.voltage_pts,
                }
        s.addsweepparameter("sweep", para)
        res1 = {"Name": "neff",
               "Result": f"::model::FDE::data::mode{self.recipe_setup.mode_simulation_setup.target_mode}::neff"}
        res2 = {"Name": "swn",
                "Result": "::model::FDE::data::frequencysweep::neff sweep"}
        s.addsweepresult("sweep", res1)
        s.addsweepresult("sweep", res2)
        s.save()

        # Get voltage index when voltage is 0V and use this point to calibrate the phase
        # and loss change. At 0V, zero phase and loss change should occur.
        voltages = np.linspace(self.recipe_setup.design_intent.voltage_start,
                               self.recipe_setup.design_intent.voltage_stop,
                               self.recipe_setup.design_intent.voltage_pts)
        index_at_zero_volt = np.argmin(np.abs(voltages))

        # Set base waveguide model to 0V bias
        s.setnamed("::model::np density", f"V_{self.recipe_setup.design_intent.contact2_name}_index", index_at_zero_volt + 1)

        # Run MODE sim
        s.mesh()
        s.findmodes()

        s.selectmode(self.recipe_setup.mode_simulation_setup.target_mode)
        s.setanalysis('track selected mode', 1);
        s.setanalysis('number of points', self.recipe_setup.mode_simulation_setup.wavl_pts);
        s.setanalysis('number of test modes', self.recipe_setup.mode_simulation_setup.num_modes);
        s.setanalysis('detailed dispersion calculation', 1);
        s.frequencysweep()

        # Save base waveguide profile
        self.recipe_results.waveguide_profile_path = mode_sim.simulation_dirpath / "waveguide.ldf"
        s.savedcard(str(self.recipe_results.waveguide_profile_path.resolve()),
                  "::model::FDE::data::frequencysweep");

        # Run voltage sweep
        s.runsweep("sweep")

        # Extract voltage vs. neff and loss results
        V = np.linspace(self.recipe_setup.design_intent.voltage_start,
                        self.recipe_setup.design_intent.voltage_stop,
                        self.recipe_setup.design_intent.voltage_pts)
        neff = s.getsweepdata("sweep", "neff")

        self.recipe_results.neff_vs_voltage = pd.DataFrame({"voltage": list(V),
                                                            "neff_r": list(np.real(neff[:,0])),
                                                            "neff_i": list(np.imag(neff[:,0]))})

        self.recipe_results.delta_neff_vs_voltage = pd.DataFrame({"voltage": list(V),
                                                                "neff_r": list(np.real(neff[:,0] - neff[index_at_zero_volt,0])),
                                                                "neff_i": list(np.imag(neff[:,0] - neff[index_at_zero_volt,0]))})

        dneff_per_cm = np.real(neff[:,0] - neff[index_at_zero_volt]) * \
                2 / (self.recipe_setup.mode_simulation_setup.wavl * um) * 1e-2
        loss_dB_per_cm = -.20 * np.log10(np.exp(-2 * np.pi * np.imag(neff[:,0]) /
                                                (self.recipe_setup.mode_simulation_setup.wavl * um)))
        self.recipe_results.phase_loss_vs_voltage = pd.DataFrame({"voltage": list(V),
                                                            "phase_per_cm": list(dneff_per_cm),
                                                            "loss_dB_per_cm": list(loss_dB_per_cm)})

        return True

class PNMicroringModulatorRecipe(DesignRecipe):
    """
    PN microring modulator (MRM) recipe used to characterize optical and electrical
    characteristics of the MRM.

                        Top View of Double Bus Ring Modulator
              DROP  ────────────────────────────────────  ADD
              PORT  ────────────────────────────────────  PORT
                                  ────────
                                /   ────   \
                               /  /      \  \
                              /  /        \  \
                             /  /          \  \
                            │  │            │  │
                            │  │            │  │
                            │  │            │  │
                            │  │            │  │
                            │  │            │  │
                             \  \          /  /
                              \  \        /  /
                               \  \      /  /
                                \   ────   /
                                  ────────
              IN    ────────────────────────────────────  THRU
             PORT   ────────────────────────────────────  PORT

    Attributes:
        recipe_setup:
            pn_design_intent: PN junction design intent
            mode_simulation_setup: MODE simulation setup
            mode_convergence_setup: MODE convergence setup
            charge_simulation_setup: CHARGE simulation setup
            charge_convergence_setup: CHARGE convergence setup
            fdtd_simulation_setup: FDTD simulation setup
            fdtd_convergence_setup: FDTD convergence setup
            interconnect_simulation_setup: INTERCONNECT simulation setup
        recipe_results:
            thru_port_data: THRU port optical response
            drop_port_data: DROP port optical response
            add_port_data: ADD port optical response
            free_spectral_range: Free spectral range calculated across wavelength range
            resonances: Resonance wavelengths and powers across wavelength range
            spectrum_vs_voltage: Optical spectrum vs. applied voltage on PN phaseshifters
            resonance_vs_voltage: Resonance wavelengths and powers vs. applied voltage
                on PN phaseshifters
    """
    def __init__(
        self,
        component: Component | None = None,
        layer_stack: LayerStack | None = None,
        process: tuple | None = None,
        pn_design_intent: PNJunctionDesignIntent | None = None,
        mode_simulation_setup: SimulationSettingsLumericalMode | None = LUMERICAL_MODE_SIMULATION_SETTINGS,
        mode_convergence_setup: ConvergenceSettingsLumericalMode | None = LUMERICAL_MODE_CONVERGENCE_SETTINGS,
        charge_simulation_setup: SimulationSettingsLumericalCharge | None = LUMERICAL_CHARGE_SIMULATION_SETTINGS,
        charge_convergence_setup: ConvergenceSettingsLumericalCharge | None = LUMERICAL_CHARGE_CONVERGENCE_SETTINGS,
        fdtd_simulation_setup: SimulationSettingsLumericalFdtd | None = SIMULATION_SETTINGS_LUMERICAL_FDTD,
        fdtd_convergence_setup: ConvergenceSettingsLumericalFdtd | None = LUMERICAL_FDTD_CONVERGENCE_SETTINGS,
        interconnect_simulation_setup: SimulationSettingsLumericalInterconnect | None = LUMERICAL_INTERCONNECT_SIMULATION_SETTINGS,
        dependencies: list[DesignRecipe] | None = None,
        dirpath: Path | None = None,
    ):
        # Extract child components (coupler and pn phaseshifter)
        coupler_component = component.insts._insts[0].cell
        pn_junction_component = component.insts._insts[2].cell
        # Set defaults
        layer_stack = layer_stack or get_layer_stack()
        pn_design_intent = pn_design_intent or PNJunctionDesignIntent()
        dependencies = dependencies or [PNJunctionRecipe(component=pn_junction_component,
                                         layer_stack=layer_stack,
                                         process=process,
                                         design_intent=pn_design_intent,
                                         mode_simulation_setup=mode_simulation_setup,
                                         mode_convergence_setup=mode_convergence_setup,
                                         charge_simulation_setup=charge_simulation_setup,
                                         charge_convergence_setup=charge_convergence_setup,
                                         dirpath=dirpath),
                        FdtdRecipe(component=coupler_component,
                                   layer_stack=layer_stack,
                                   process=process,
                                   simulation_setup=fdtd_simulation_setup,
                                   convergence_setup=fdtd_convergence_setup,
                                   dirpath=dirpath)]

        super().__init__(cell=component,
                         dependencies=dependencies,
                         layer_stack=layer_stack,
                         process=process,
                         dirpath=dirpath)

        self.recipe_setup.pn_design_intent = pn_design_intent
        self.recipe_setup.mode_simulation_setup = mode_simulation_setup
        self.recipe_setup.mode_convergence_setup = mode_convergence_setup
        self.recipe_setup.charge_simulation_setup = charge_simulation_setup
        self.recipe_setup.charge_convergence_setup = charge_convergence_setup
        self.recipe_setup.fdtd_simulation_setup = fdtd_simulation_setup
        self.recipe_setup.fdtd_convergence_setup = fdtd_convergence_setup
        self.recipe_setup.interconnect_simulation_setup = interconnect_simulation_setup

    @eval_decorator
    def eval(self, run_convergence: bool = True) -> bool:
        """
        Run PN microring modulator recipe to characterize microring

        1. Extracts electro-optic characteristics of PN phaseshifter
        2. Extracts s-parameters of coupler
        3. Combines phaseshifter and coupler in INTERCONNECT  sim to characterize
        ring

        Parameters:
            run_convergence: Run convergence if True

        Returns:
            success: True if recipe completed properly
        """
        pn_recipe = self.dependencies.constituent_recipes[0]
        coupler_recipe = self.dependencies.constituent_recipes[1]

        # Create waveguide compact model representing PN junction waveguide
        waveguide_model = WAVEGUIDE_COMPACT_MODEL.copy()
        waveguide_model.settings.update({
            "name": "WAVEGUIDE",
            "ldf filename": str(pn_recipe.recipe_results.waveguide_profile_path.resolve()),
            "length": self.cell.settings.length_pn * um,
        })
        create_compact_model(model=waveguide_model, dirpath=self.recipe_dirpath)
        self.recipe_results.waveguide_model_settings = waveguide_model

        # Create phaseshifter compact model
        ps_model = PHASESHIFTER_COMPACT_MODEL.copy()
        ps_model.settings.update({
            "name": "PHASESHIFTER",
            "frequency": speed_of_light / (self.recipe_setup.mode_simulation_setup.wavl * um),
            "length": self.cell.settings.length_pn * um,
            "load from file": False,
            "measurement type": "effective index",
            "measurement": np.array(pn_recipe.recipe_results.delta_neff_vs_voltage),
        })
        create_compact_model(model=ps_model, dirpath=self.recipe_dirpath)
        self.recipe_results.phaseshifter_model_settings = ps_model

        # Create coupler compact model
        coupler_model = SPARAM_COMPACT_MODEL.copy()
        coupler_model.settings.update({
            "name": "COUPLER",
            "s parameters filename": str(coupler_recipe.recipe_results.filepath_dat.resolve()),
        })
        create_compact_model(model=coupler_model, dirpath=self.recipe_dirpath)
        self.recipe_results.coupler_model_settings = coupler_model

        # Create ring modulator in INTERCONNECT
        # characteristics
        import lumapi
        s = lumapi.INTERCONNECT(hide=False)
        s.loadcustom(str(self.recipe_dirpath.resolve()))

        wg1 = s.addelement(f"::custom::{waveguide_model.settings['name'].lower()}")
        wg2 = s.addelement(f"::custom::{waveguide_model.settings['name'].lower()}")
        ps1 = s.addelement(f"::custom::{ps_model.settings['name'].lower()}")
        ps2 = s.addelement(f"::custom::{ps_model.settings['name'].lower()}")
        cp1 = s.addelement(f"::custom::{coupler_model.settings['name'].lower()}")
        cp2 = s.addelement(f"::custom::{coupler_model.settings['name'].lower()}")
        dc1 = s.addelement("DC Source")
        dc2 = s.addelement("DC Source")

        s.connect(cp1.name, "o3", wg1.name, "port 1")
        s.connect(wg1.name, "port 2", ps1.name, "port 1")
        s.connect(ps1.name, "port 2", cp2.name, "o2")
        s.connect(cp2.name, "o3", wg2.name, "port 1")
        s.connect(wg2.name, "port 2", ps2.name, "port 1")
        s.connect(ps2.name, "port 2", cp1.name, "o2")

        s.connect(dc1.name, "output", ps1.name, "modulation")
        s.connect(dc2.name, "output", ps2.name, "modulation")

        osa = s.addelement("Optical Network Analyzer", properties={
            "number of input ports": 3,
            "input parameter": "start and stop",
            "start frequency": speed_of_light / (self.recipe_setup.interconnect_simulation_setup.wavl_start * um),
            "stop frequency": speed_of_light / (self.recipe_setup.interconnect_simulation_setup.wavl_end * um),
            "number of points": self.recipe_setup.interconnect_simulation_setup.wavl_pts,
        })
        s.connect(osa.name, "output", cp1.name, "o1")
        s.connect(osa.name, "input 1", cp1.name, "o4")
        s.connect(osa.name, "input 2", cp2.name, "o1")
        s.connect(osa.name, "input 3", cp2.name, "o4")
        s.save(str(self.recipe_dirpath / f"{self.cell.name}.icp"))
        self.interconnect_session = s

        # Run simulation and extract static response MRM figures of merit
        s.run()
        s.select(osa.name)

        # Get THRU port response
        data = s.getresult(osa.name, "input 1/mode 1/gain")
        self.recipe_results.thru_port_data = pd.DataFrame({"wavelength": data["wavelength"][:,0] / um,
                                  "gain": data["mode 1 gain (dB)"]})

        # Get DROP port response
        data = s.getresult(osa.name, "input 3/mode 1/gain")
        self.recipe_results.drop_port_data = pd.DataFrame({"wavelength": data["wavelength"][:, 0] / um,
                                  "gain": data["mode 1 gain (dB)"]})

        # Get ADD port response
        data = s.getresult(osa.name, "input 2/mode 1/gain")
        self.recipe_results.add_port_data = pd.DataFrame(
            {"wavelength": data["wavelength"][:, 0] / um,
             "gain": data["mode 1 gain (dB)"]})

        self.recipe_results.free_spectral_range = get_free_spectral_range(wavelength=list(self.recipe_results.thru_port_data.loc[:,"wavelength"]),
                                                                          power=list(self.recipe_results.thru_port_data.loc[:,"gain"]),
                                                                          peaks_flipped=True)
        self.recipe_results.resonances = get_resonances(wavelength=list(self.recipe_results.thru_port_data.loc[:,"wavelength"]),
                                                      power=list(self.recipe_results.thru_port_data.loc[:,"gain"]),
                                                      peaks_flipped=True)


        # Sweep voltage bias and view resonance change
        voltages = np.linspace(self.recipe_setup.pn_design_intent.voltage_start,
                               self.recipe_setup.pn_design_intent.voltage_stop,
                               self.recipe_setup.pn_design_intent.voltage_pts)
        spectrums = []
        wavelength_resonances = []
        power_resonances = []
        for voltage in voltages:
            s.switchtodesign()
            s.setnamed(dc1.name, "amplitude", voltage)
            s.setnamed(dc2.name, "amplitude", voltage)
            s.run()

            # Get optical spectrum and resonances
            data1 = s.getresult(osa.name, "input 1/mode 1/gain")
            data2 = s.getresult(osa.name, "input 3/mode 1/gain")
            spectrums.append(pd.DataFrame({
                "wavelength": data1["wavelength"][:, 0] / um,
                "thru": data1["mode 1 gain (dB)"],
                "drop": data2["mode 1 gain (dB)"]
            }))
            resonance = get_resonances(wavelength=list(data1["wavelength"][:, 0] / um),
                                          power=list(data1["mode 1 gain (dB)"]),
                                          peaks_flipped=True)
            wavelength_resonances.append(list(resonance.loc[:, "resonant_wavelength"]))
            power_resonances.append(list(resonance.loc[:, "resonant_power"]))

        self.recipe_results.spectrum_vs_voltage = pd.DataFrame({"spectrum": spectrums,
                                                                "voltage": voltages})
        self.recipe_results.resonance_vs_voltage = pd.DataFrame({"wavelength_resonances": wavelength_resonances,
                                                                 "power_resonances": power_resonances,
                                                                 "voltage": voltages})

        return True

