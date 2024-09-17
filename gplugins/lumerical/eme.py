"""
Lumerical EME Plugin

Author: Sean Lam
Contact: seanl@ece.ubc.ca
"""

from __future__ import annotations

from pathlib import Path

import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gdsfactory.component import Component
from gdsfactory import logger
from gdsfactory.pdk import get_layer_stack
from gdsfactory.technology.layer_stack import LayerStack
from gdsfactory.typings import PathType

from gplugins.lumerical.config import marker_list, um
from gplugins.lumerical.convergence_settings import (
    LUMERICAL_EME_CONVERGENCE_SETTINGS,
    ConvergenceSettingsLumericalEme,
)
from gplugins.lumerical.simulation_settings import (
    LUMERICAL_EME_SIMULATION_SETTINGS,
    SimulationSettingsLumericalEme,
)
from gplugins.lumerical.utils import Simulation, draw_geometry, layerstack_to_lbr


class LumericalEmeSimulation(Simulation):
    """
    Lumerical EME simulation plugin for running EME simulations on GDSFactory designs

    Attributes:
        component: Component geometry to simulate
        layerstack: PDK layerstack
        session: Lumerical session
        simulation_settings: EME simulation settings
        convergence_settings: EME convergence settings
        dirpath: Directory where simulation files are saved
        convergence_results: Dynamic object used to store convergence results
            simulation_settings: EME simulation settings
            convergence_settings: EME convergence settings
            layerstack: PDK layerstack
            component_hash: Component geometry hash
            mesh_convergence_data: Mesh convergence results
            cell_convergence_data: Cell convergence results
            mode_convergence_data: Mode convergence results
            overall_convergence_data: Combination of mesh and cell convergence results

    """

    def __init__(
        self,
        component: Component,
        layerstack: LayerStack | None = None,
        session: object | None = None,
        simulation_settings: SimulationSettingsLumericalEme = LUMERICAL_EME_SIMULATION_SETTINGS,
        convergence_settings: ConvergenceSettingsLumericalEme = LUMERICAL_EME_CONVERGENCE_SETTINGS,
        dirpath: PathType | None = "",
        hide: bool = True,
        run_mesh_convergence: bool = False,
        run_cell_convergence: bool = False,
        run_mode_convergence: bool = False,
        run_overall_convergence: bool = False,
        override_convergence: bool = False,
        **settings,
    ):
        """
        Set up EME simulation based on component geometry and simulation settings. Optionally, run convergence.

        Parameters:
            component: Component geometry to simulate
            layerstack: PDK layerstack
            session: Lumerical session
            simulation_settings: EME simulation settings
            convergence_settings: EME convergence settings
            dirpath: Root directory where simulations are saved. A sub-directory labeled with the class name and hash is
                    be where simulation files are saved.
            hide: Hide simulation if True, else show GUI
            run_mesh_convergence: If True, run sweep of mesh and monitor sparam convergence.
            run_cell_convergence: If True, run sweep of number of cells in central group span and monitor sparam convergence.
            run_mode_convergence: If True, run sweep of number of modes and monitor sparam convergence.
            run_overall_convergence: If True, run combination of mesh and cell convergence to quicken convergence.
            override_convergence: Override convergence results and run convergence testing
        """
        # Set up variables
        dirpath = dirpath or Path(".")
        simulation_settings = dict(simulation_settings)

        if hasattr(component.info, "simulation_settings"):
            simulation_settings |= component.info.simulation_settings
            logger.info(
                f"Updating {component.name!r} sim settings {component.simulation_settings}"
            )
        for setting in settings:
            if setting not in simulation_settings:
                raise ValueError(
                    f"Invalid setting {setting!r} not in ({list(simulation_settings.keys())})"
                )

        simulation_settings.update(**settings)
        ss = SimulationSettingsLumericalEme(**simulation_settings)

        # Check number of cell groups are aligned
        if not (len(ss.group_cells) == len(ss.group_subcell_methods)):
            raise ValueError(
                "Number of cell groups are not aligned.\n"
                + f"Group Cells ({len(ss.group_cells)}): {ss.group_cells}\n"
                + f"Group Subcell Methods ({len(ss.group_subcell_methods)}): {ss.group_subcell_methods}"
            )

        layerstack = layerstack or get_layer_stack()

        # Save instance variables
        self.simulation_settings = ss
        self.convergence_settings = convergence_settings
        self.component = component
        self.layerstack = layerstack
        self.dirpath = dirpath

        # Initialize parent class
        super().__init__(
            component=self.component,
            layerstack=self.layerstack,
            simulation_settings=self.simulation_settings,
            convergence_settings=self.convergence_settings,
            dirpath=self.dirpath,
        )

        # If convergence data is already available, update simulation settings
        if (
            self.convergence_is_fresh()
            and self.convergence_results.available()
            and not override_convergence
        ):
            try:
                self.load_convergence_results()
                # Check if convergence settings, component, and layerstack are the same. If the same, use the simulation settings from file. Else,
                # run convergence testing by overriding convergence results. This covers any collisions in hashes.
                if self.is_same_convergence_results():
                    self.convergence_settings = (
                        self.convergence_results.convergence_settings
                    )
                    self.simulation_settings = (
                        ss
                    ) = self.convergence_results.simulation_settings
                    # Update hash since settings have changed
                    self.last_hash = hash(self)
                else:
                    override_convergence = True
            except (AttributeError, FileNotFoundError) as err:
                logger.warning(f"{err}\nRun convergence.")
                override_convergence = True

        # Set up EME simulation based on provided simulation settings
        if not session:
            try:
                import lumapi
            except Exception as e:
                logger.error(
                    "Cannot import lumapi (Python Lumerical API). "
                    "You can add set the PYTHONPATH variable or add it with `sys.path.append()`"
                )
                raise e
            session = lumapi.MODE(hide=hide)
        self.session = session
        s = session
        s.newproject()

        ports = component.get_ports_list(port_type="optical")
        if not ports:
            raise ValueError(f"{component.name!r} does not have any optical ports")
        if len(ports) > 2:
            raise ValueError(
                f"{component.name!r} has more than 2 ports. EME only supports 2 port devices."
            )

        # Extend component ports beyond simulation region
        component_with_booleans = layerstack.get_component_with_derived_layers(
            component
        )
        component_with_padding = gf.add_padding_container(
            component_with_booleans, default=0
        )

        component_extended = gf.components.extend_ports(
            component_with_padding, length=ss.port_extension
        )

        component_extended_beyond_pml = gf.components.extension.extend_ports(
            component=component_extended, length=ss.port_extension
        )
        component_extended_beyond_pml.name = "top"
        gdspath = component_extended_beyond_pml.write_gds()

        process_file_path = layerstack_to_lbr(
            ss.material_name_to_lumerical, layerstack, self.simulation_dirpath.resolve()
        )

        # Create device geometry
        draw_geometry(s, gdspath, process_file_path)

        # Fit material models
        for layer_name in layerstack.to_dict():
            s.select("layer group")
            material_name = s.getlayer(layer_name, "pattern material")
            try:
                s.setmaterial(material_name, "wavelength min", ss.wavelength_start * um)
                s.setmaterial(material_name, "wavelength max", ss.wavelength_stop * um)
                s.setmaterial(material_name, "tolerance", ss.material_fit_tolerance)
            except Exception:
                logger.warning(
                    f"Material {material_name} cannot be found in database, skipping material fit."
                )

        # Create simulation region
        x_min = component_extended.xmin * um
        x_max = component_extended.xmax * um
        y_min = (component_extended.ymin - ss.ymargin) * um
        y_max = (component_extended.ymax + ss.ymargin) * um

        layer_to_thickness = layerstack.get_layer_to_thickness()
        layer_to_zmin = layerstack.get_layer_to_zmin()
        layers_thickness = [
            layer_to_thickness[layer]
            for layer in component_with_booleans.get_layers()
            if layer in layer_to_thickness
        ]
        if not layers_thickness:
            raise ValueError(
                f"no layers for component {component.get_layers()}"
                f"in layer stack {layerstack}"
            )
        layers_zmin = [
            layer_to_zmin[layer]
            for layer in component_with_booleans.get_layers()
            if layer in layer_to_zmin
        ]
        component_thickness = max(layers_thickness)
        component_zmin = min(layers_zmin)

        z = (component_zmin + component_thickness) / 2 * um
        z_span = (2 * ss.zmargin + component_thickness) * um

        s.addeme()
        s.set("display cells", 1)
        s.set("x min", x_min)
        s.set("y min", y_min)
        s.set("y max", y_max)
        s.set("z", z)
        s.set("z span", z_span)

        s.set("wavelength", ss.wavelength * um)
        s.setemeanalysis("Wavelength sweep", 1)
        s.setemeanalysis("start wavelength", ss.wavelength_start)
        s.setemeanalysis("stop wavelength", ss.wavelength_stop)

        s.set("number of cell groups", len(ss.group_cells))
        s.set("cells", np.array(ss.group_cells))

        # Use component bounds for the group spans
        group_spans = []
        mid_span = (x_max - x_min - 2 * ss.port_extension * um) / (
            len(ss.group_cells) - 2
        )
        for i in range(0, len(ss.group_cells)):
            if i == 0 or i == len(ss.group_cells) - 1:
                group_spans.append(ss.port_extension * um)
            else:
                group_spans.append(mid_span)
        group_spans = np.array(group_spans)

        s.set("group spans", group_spans)

        # Convert subcell methods to int for Lumerical to interpret
        # 1 = CVCS, 0 = none
        subcell_methods = []
        for method in ss.group_subcell_methods:
            if method == "CVCS":
                subcell_methods.append(1)
            else:
                subcell_methods.append(0)
        s.set("subcell method", np.array(subcell_methods))

        s.set("number of modes for all cell groups", ss.num_modes)
        s.set("energy conservation", ss.energy_conservation)

        s.set("define y mesh by", "maximum mesh step")
        s.set("define z mesh by", "maximum mesh step")

        s.set("dy", ss.wavelength / ss.mesh_cells_per_wavelength * um)
        s.set("dz", ss.wavelength / ss.mesh_cells_per_wavelength * um)

        s.set("y min bc", ss.ymin_boundary)
        s.set("y max bc", ss.ymax_boundary)
        s.set("z min bc", ss.zmin_boundary)
        s.set("z max bc", ss.zmax_boundary)

        s.set("pml layers", ss.pml_layers)

        s.save(str(self.simulation_dirpath.resolve() / f"{component.name}.lms"))

        # Run convergence testing if no convergence results are available or user wants to override convergence results
        # or if setup has changed
        if (
            not self.convergence_results.available()
            or override_convergence
            or not self.convergence_is_fresh()
        ):
            if run_overall_convergence:
                if not hide:
                    logger.info("Running overall convergence")
                self.convergence_results.overall_convergence_data = (
                    self.update_overall_convergence(plot=not hide)
                )
            else:
                if run_mesh_convergence:
                    if not hide:
                        logger.info("Running mesh convergence.")
                    self.convergence_results.mesh_convergence_data = (
                        self.update_mesh_convergence(plot=not hide)
                    )

                if run_cell_convergence:
                    if not hide:
                        logger.info("Running cell convergence.")
                    self.convergence_results.cell_convergence_data = (
                        self.update_cell_convergence(plot=not hide)
                    )

            if run_mode_convergence:
                if not hide:
                    logger.info("Running mode convergence.")
                self.convergence_results.mode_convergence_data = (
                    self.update_mode_convergence(plot=not hide)
                )

            if (run_overall_convergence and run_mode_convergence) or (
                run_mesh_convergence and run_cell_convergence and run_mode_convergence
            ):
                # Save setup and results for convergence
                self.save_convergence_results()
                if not hide:
                    logger.info("Saved convergence results.")

        if not hide:
            plt.show()

    def update_mesh_convergence(self, plot: bool = False) -> pd.DataFrame:
        """
        Update simulation based on mesh convergence testing. Updates both Lumerical session and simulation settings.

        Parameters:
            plot: Plot and save convergence results

        Returns:
            Convergence results
        """

        s = self.session
        cs = self.convergence_settings
        ss = self.simulation_settings

        s21 = []
        s11 = []
        mesh_cells_per_wavl = []
        converged = False
        while not converged:
            s.switchtolayout()
            s.set("dy", ss.wavelength / ss.mesh_cells_per_wavelength * um)
            s.set("dz", ss.wavelength / ss.mesh_cells_per_wavelength * um)
            # Get sparams
            s.run()
            s.emepropagate()
            S = s.getresult("EME", "user s matrix")
            s11.append(abs(S[0, 0]) ** 2)
            s21.append(abs(S[1, 0]) ** 2)
            mesh_cells_per_wavl.append(ss.mesh_cells_per_wavelength)

            # Refine mesh
            ss.mesh_cells_per_wavelength += 1

            # Check whether convergence has been reached
            if len(s21) > cs.passes or len(s11) > cs.passes:
                # Calculate maximum diff in sparams
                sparam_diff = max(
                    [
                        max(abs(s21[-1] - np.array(s21[-(cs.passes + 1) : -1]))),
                        max(abs(s11[-1] - np.array(s11[-(cs.passes + 1) : -1]))),
                    ]
                )
                if plot:
                    logger.info(
                        f"Mesh cells per wavelength: {ss.mesh_cells_per_wavelength} | S-Param Diff: {sparam_diff}"
                    )
                if sparam_diff < cs.sparam_diff:
                    converged = True
                else:
                    converged = False

        if plot:
            plt.figure()
            plt.plot(mesh_cells_per_wavl, s21)
            plt.plot(mesh_cells_per_wavl, s11)
            plt.legend(["|S21|^2", "|S11|^2"])
            plt.grid("on")
            plt.xlabel("Mesh Cells Per Wavelength")
            plt.ylabel("Magnitude")
            plt.title(f"Mesh Convergence | Wavelength={ss.wavelength}um")
            plt.savefig(
                str(
                    self.simulation_dirpath.resolve()
                    / f"{self.component.name}_eme_mesh_convergence.png"
                )
            )

        convergence_data = pd.DataFrame.from_dict(
            {"mesh_cells": mesh_cells_per_wavl, "s21": list(s21), "s11": list(s11)}
        )
        convergence_data.to_csv(
            str(
                self.simulation_dirpath.resolve()
                / f"{self.component.name}_eme_mesh_convergence.csv"
            )
        )
        return convergence_data

    def update_cell_convergence(self, plot: bool = False) -> pd.DataFrame:
        """
        Update simulation based on cell convergence testing (number of slices across the device center).
        Updates both Lumerical session and simulation settings.

        Parameters:
            plot: Plot and save convergence results

        Returns:
            Convergence results
        """
        s = self.session
        cs = self.convergence_settings
        ss = self.simulation_settings

        s21 = []
        s11 = []
        num_cells = []
        converged = False
        while not converged:
            s.switchtolayout()
            s.setnamed("EME", "cells", np.array(ss.group_cells))
            s.run()
            s.emepropagate()
            S = s.getresult("EME", "user s matrix")
            s11.append(abs(S[0, 0]) ** 2)
            s21.append(abs(S[1, 0]) ** 2)
            num_cells.append(ss.group_cells[1:-1])

            for i in range(1, len(ss.group_cells) - 1):
                ss.group_cells[i] += 1

            # Check whether convergence has been reached
            if len(s21) > cs.passes or len(s11) > cs.passes:
                # Calculate maximum diff in sparams
                sparam_diff = max(
                    [
                        max(abs(s21[-1] - np.array(s21[-(cs.passes + 1) : -1]))),
                        max(abs(s11[-1] - np.array(s11[-(cs.passes + 1) : -1]))),
                    ]
                )
                if plot:
                    logger.info(
                        f"Num cells: {ss.group_cells} | S-Param Diff: {sparam_diff}"
                    )
                if sparam_diff < cs.sparam_diff:
                    converged = True
                else:
                    converged = False

        num_cells = np.array(num_cells)
        if plot:
            plt.figure()
            plt.plot(list(num_cells[:, 0]), s21)
            plt.plot(list(num_cells[:, 0]), s11)
            plt.xticks(list(num_cells[:, 0]), [f"{list(row)}" for row in num_cells])
            plt.setp(plt.xticks()[1], rotation=75, horizontalalignment="center")
            plt.legend(["|S21|^2", "|S11|^2"])
            plt.grid("on")
            plt.xlabel("Number of Cells")
            plt.ylabel("Magnitude")
            plt.title(f"Cell Convergence | Wavelength={ss.wavelength}um")
            plt.tight_layout()
            plt.savefig(
                str(
                    self.simulation_dirpath.resolve()
                    / f"{self.component.name}_eme_cell_convergence.png"
                )
            )

        convergence_data = pd.DataFrame.from_dict(
            {"num_cells": list(num_cells[:, 0]), "s21": list(s21), "s11": list(s11)}
        )
        convergence_data.to_csv(
            str(
                self.simulation_dirpath.resolve()
                / f"{self.component.name}_eme_cell_convergence.csv"
            )
        )
        return convergence_data

    def update_mode_convergence(self, plot: bool = False) -> pd.DataFrame:
        """
        Update simulation based on mode convergence testing (number of modes required to be accurate).
        Updates both Lumerical session and simulation settings.

        Parameters:
            plot: Plot and save convergence results

        Returns:
            Convergence results
        """
        s = self.session
        cs = self.convergence_settings
        ss = self.simulation_settings

        converged = False
        while not converged:
            s.switchtolayout()
            s.setnamed("EME", "number of modes for all cell groups", ss.num_modes)
            s.run()
            # s.emepropagate()

            s.setemeanalysis("Mode convergence sweep", 1)
            s.emesweep("mode convergence sweep")

            # get mode convergence sweep result
            S = s.getemesweep("S_mode_convergence_sweep")

            # plot S21 vs number of modes
            s21 = abs(S["s21"]) ** 2
            s11 = abs(S["s11"]) ** 2
            modes = S["modes"]

            # Check whether convergence has been reached
            if len(s21) > cs.passes or len(s11) > cs.passes:
                # Calculate maximum diff in sparamsgd
                sparam_diff = max(
                    [
                        max(abs(s21[-1] - np.array(s21[-(cs.passes + 1) : -1]))),
                        max(abs(s11[-1] - np.array(s11[-(cs.passes + 1) : -1]))),
                    ]
                )
                if plot:
                    logger.info(
                        f"Num modes: {ss.num_modes} | S-Param Diff: {sparam_diff}"
                    )
                if sparam_diff < cs.sparam_diff:
                    converged = True
                    break
                else:
                    converged = False

            ss.num_modes += 5

        if plot:
            plt.figure()
            plt.plot(modes, s21)
            plt.plot(modes, s11)
            plt.legend(["|S21|^2", "|S11|^2"])
            plt.grid("on")
            plt.xlabel("Number of Modes")
            plt.ylabel("Magnitude")
            plt.title(f"Mode Convergence | Wavelength={ss.wavelength}um")
            plt.tight_layout()
            plt.savefig(
                str(
                    self.simulation_dirpath.resolve()
                    / f"{self.component.name}_eme_mode_convergence.png"
                )
            )

        convergence_data = pd.DataFrame.from_dict(
            {"modes": list(modes), "s21": list(s21), "s11": list(s11)}
        )
        convergence_data.to_csv(
            str(
                self.simulation_dirpath.resolve()
                / f"{self.component.name}_eme_mode_convergence.csv"
            )
        )
        return convergence_data

    def update_overall_convergence(self, plot: bool = False) -> pd.DataFrame:
        """
        Runs mesh and cell convergence simultaneously, switching strategies every time sparam variation increases.

        Parameters:
            plot: Plot and save convergence results

        Returns:
            Convergence data
            | mesh_cells | num_cells | s21   | s11   |
            | int        | int       | float | float |
        """
        s = self.session
        cs = self.convergence_settings
        ss = self.simulation_settings

        s21 = []
        s11 = []
        mesh_cells_per_wavl = []
        num_cells = []
        sparam_diffs = []
        converged = False
        algo = True
        while not converged:
            s.switchtolayout()
            s.set("dy", ss.wavelength / ss.mesh_cells_per_wavelength * um)
            s.set("dz", ss.wavelength / ss.mesh_cells_per_wavelength * um)
            s.setnamed("EME", "cells", np.array(ss.group_cells))
            # Get sparams and refine mesh
            s.run()
            s.emepropagate()
            S = s.getresult("EME", "user s matrix")
            s11.append(abs(S[0, 0]) ** 2)
            s21.append(abs(S[1, 0]) ** 2)
            mesh_cells_per_wavl.append(ss.mesh_cells_per_wavelength)
            num_cells.append(ss.group_cells[1:-1])

            # Check whether convergence has been reached
            if len(s21) > cs.passes or len(s11) > cs.passes:
                # Calculate maximum diff in sparams
                sparam_diff = max(
                    [
                        max(abs(s21[-1] - np.array(s21[-(cs.passes + 1) : -1]))),
                        max(abs(s11[-1] - np.array(s11[-(cs.passes + 1) : -1]))),
                    ]
                )
                # If sparam diff is decreasing, continue to use existing convergence algorithm
                # Else, switch algorithms
                sparam_diffs.append(sparam_diff)
                if len(sparam_diffs) > 1:
                    if (sparam_diffs[-1] - sparam_diffs[-2]) > 0:
                        algo = not algo

                    if algo:
                        ss.mesh_cells_per_wavelength += 1
                    else:
                        for i in range(1, len(ss.group_cells) - 1):
                            ss.group_cells[i] += 1
                else:
                    ss.mesh_cells_per_wavelength += 1
                    for i in range(1, len(ss.group_cells) - 1):
                        ss.group_cells[i] += 1

                # Print convergence step
                if plot:
                    logger.info(
                        f"Mesh cells per wavelength: {ss.mesh_cells_per_wavelength} | Num cells: {ss.group_cells} | S-Param Diff: {sparam_diff}"
                    )

                # Check for convergence
                if sparam_diff < cs.sparam_diff:
                    converged = True
                else:
                    converged = False
            else:
                # Update mesh and number of cells
                ss.mesh_cells_per_wavelength += 1
                for i in range(1, len(ss.group_cells) - 1):
                    ss.group_cells[i] += 1

        if plot:
            plt.figure()
            plt.plot(mesh_cells_per_wavl, s21)
            plt.plot(mesh_cells_per_wavl, s11)
            plt.xticks(
                mesh_cells_per_wavl,
                [
                    f"Mesh Cells: {mesh_cells_per_wavl[i]} | Num Cells: {num_cells[i]}"
                    for i in range(0, len(mesh_cells_per_wavl))
                ],
            )
            plt.setp(plt.xticks()[1], rotation=75, horizontalalignment="center")
            plt.legend(["|S21|^2", "|S11|^2"])
            plt.grid("on")
            plt.xlabel("Mesh Cells Per Wavelength | Num Cells")
            plt.ylabel("Magnitude")
            plt.title(f"Overall Convergence | Wavelength={ss.wavelength}um")
            plt.tight_layout()
            plt.savefig(
                str(
                    self.simulation_dirpath.resolve()
                    / f"{self.component.name}_eme_overall_convergence.png"
                )
            )

        convergence_data = pd.DataFrame.from_dict(
            {
                "mesh_cells": mesh_cells_per_wavl,
                "num_cells": num_cells,
                "s21": list(s21),
                "s11": list(s11),
            }
        )
        convergence_data.to_csv(
            str(
                self.simulation_dirpath.resolve()
                / f"{self.component.name}_eme_overall_convergence.csv"
            )
        )
        return convergence_data

    def get_length_sweep(
        self,
        start_length: float = 1,
        stop_length: float = 100,
        num_pts: int = 100,
        group: int = 1,
    ) -> pd.DataFrame:
        """
        Get length sweep sparams.

        Parameters:
            start_length: Start length (um)
            stop_length: Stop length (um)
            num_pts: Number of points along sweep
            group: Group span to sweep. First group starts from 0

        Returns:
            Dataframe with length sweep and complex sparams
            | length (m) | s11     | s21     | s12     | s11     |
            | float      | complex | complex | complex | complex |
        """

        s = self.session
        s.run()
        s.emepropagate()

        # set propagation sweep settings
        s.setemeanalysis("propagation sweep", 1)
        s.setemeanalysis("parameter", f"group span {group + 1}")
        s.setemeanalysis("start", start_length * um)
        s.setemeanalysis("stop", stop_length * um)
        s.setemeanalysis("number of points", num_pts)

        s.emesweep()

        S = s.getemesweep("S")

        # Get sparams
        s21 = list(S["s21"])
        s22 = list(S["s22"])
        s11 = list(S["s11"])
        s12 = list(S["s12"])
        group_span = list(S[f"group_span_{group + 1}"])

        length_sweep = pd.DataFrame.from_dict(
            {
                "length": [L[0] for L in group_span],
                "s11": s11,
                "s21": s21,
                "s12": s12,
                "s22": s22,
            }
        )

        length_sweep2 = pd.DataFrame.from_dict(
            {
                "length": [L[0] for L in group_span],
                "s11": list(abs(S["s11"]) ** 2),
                "s21": list(abs(S["s21"]) ** 2),
                "s12": list(abs(S["s12"]) ** 2),
                "s22": list(abs(S["s22"]) ** 2),
            }
        )

        length_sweep2.to_csv(
            str(self.simulation_dirpath.resolve() / f"{self.component.name}_length_sweep.csv")
        )
        return length_sweep

    def plot_length_sweep(
        self,
        start_length: float = 1,
        stop_length: float = 100,
        num_pts: int = 100,
        group: int = 1,
    ) -> None:
        """
        Plot length sweep.

        Parameters:
            start_length: Start length (um)
            stop_length: Stop length (um)
            num_pts: Number of points along sweep
            group: Group span to sweep. First group starts from 0.

        Returns:
            Figure handle
        """
        sweep_data = self.get_length_sweep(start_length, stop_length, num_pts, group)

        fig = plt.figure()
        plt.plot(sweep_data.loc[:, "length"] / um, abs(sweep_data.loc[:, "s11"]) ** 2)
        plt.plot(sweep_data.loc[:, "length"] / um, abs(sweep_data.loc[:, "s21"]) ** 2)
        plt.plot(sweep_data.loc[:, "length"] / um, abs(sweep_data.loc[:, "s12"]) ** 2)
        plt.plot(sweep_data.loc[:, "length"] / um, abs(sweep_data.loc[:, "s22"]) ** 2)
        plt.legend(["|S11|^2", "|S21|^2", "|S12|^2", "|S22|^2"])
        plt.grid("on")
        plt.xlabel("Length (um)")
        plt.ylabel("Magnitude")
        plt.title(f"Length Sweep | Wavelength={self.simulation_settings.wavelength}um")
        plt.tight_layout()
        plt.savefig(str(self.simulation_dirpath.resolve() / f"{self.component.name}_length_sweep.png"))

        return fig

    def get_mode_coupling(
        self, input_mode: int, max_coupled_mode: int = None, group: int = 1
    ) -> dict[str, pd.DataFrame]:
        """
        Get mode coupling coefficients from the input_mode to the max_coupled_mode.
        Mode numbers start from 1 (fundamental mode).

        Parameters:
            input_mode: Mode of interest. Energy couples away from this mode to other modes.
            max_coupled_mode: Maximum mode number to consider; all modes below this mode will be considered as well.
            group: Group of cells to consider. First group is 0.

        Returns:
            Dictionary of coupling coefficients vs. position along device:
            1. forward_sparam - Forward sparam coupling (dB)
            | mode1 | mode2 | ...
            | float | float | ...
            2. backward_sparam Backward sparam coupling (dB)
            | mode1 | mode2 | ...
            | float | float | ...
            3. overlap - Overlap integrals (dB)
            | mode1 | mode2 | ...
            | float | float | ...
            4. position - Position along device (m)
            | position |
            | float    |
        """

        s = self.session
        ss = self.simulation_settings
        max_coupled_mode = max_coupled_mode or ss.num_modes

        if s.layoutmode():
            s.run()
        s.emepropagate()

        # Get starting point for number of cell and group span
        cells = [int(val[0]) for val in s.getnamed("EME", "cells")]
        num_cells = cells[group]
        group_span = round(s.getnamed("EME", "group spans")[group][0], 15)
        cell_start = 0
        for i in range(0, group):
            cell_start += cells[i]

        ## Extract backward coupling coefficients
        for i in range(cell_start + 1, num_cells + cell_start + 1):
            # Get Column 0 Data: S[:,0]
            # Get Row 0 Data: S[0,:]
            S = s.getresult(f"EME::Cells::cell_{i}", f"S_{i}_{i+1}")

            # Get coupling based on S parameters and overlap coefficients
            if i == cell_start + 1:
                # Create initial column
                S_bcoupling = np.c_[
                    10 * np.log10(abs(S[0:max_coupled_mode, input_mode - 1]) ** 2)
                ]
            else:
                # Populate subsequent columns
                S_bcoupling = np.c_[
                    S_bcoupling,
                    10 * np.log10(abs(S[0:+max_coupled_mode, input_mode - 1]) ** 2),
                ]

        ## Extract forward coupling coefficients and overlap coefficients
        for i in range(cell_start + 1, num_cells + cell_start + 1):
            # Get Column 0 Data: S[:,0]
            # Get Row 0 Data: S[0,:]
            S = s.getresult(f"EME::Cells::cell_{i}", f"S_{i}_{i+1}".format(i, i + 1))
            overlap = s.getresult(f"EME::Cells::cell_{i}", f"overlap_{i}_{i+1}")

            # Get coupling based on S parameters and overlap coefficients
            if i == cell_start + 1:
                # Create initial column
                S_fcoupling = np.c_[
                    10
                    * np.log10(
                        abs(
                            S[
                                ss.num_modes : ss.num_modes + max_coupled_mode,
                                input_mode - 1,
                            ]
                        )
                        ** 2
                    )
                ]
                overlap_coupling = np.c_[
                    10 * np.log10(abs(overlap[0:max_coupled_mode, input_mode - 1]) ** 2)
                ]
            else:
                # Populate subsequent columns
                S_fcoupling = np.c_[
                    S_fcoupling,
                    10
                    * np.log10(
                        abs(
                            S[
                                ss.num_modes : ss.num_modes + max_coupled_mode,
                                input_mode - 1,
                            ]
                        )
                        ** 2
                    ),
                ]
                overlap_coupling = np.c_[
                    overlap_coupling,
                    10
                    * np.log10(abs(overlap[0:max_coupled_mode, input_mode - 1]) ** 2),
                ]

        position = np.linspace(0, group_span, num_cells + 1)[1:]

        coupling_coefficients = {
            "forward_sparam": pd.DataFrame.from_dict(
                {
                    f"mode{i+1}": list(S_fcoupling[i, :])
                    for i in range(0, S_fcoupling.shape[0])
                }
            ),
            "backward_sparam": pd.DataFrame.from_dict(
                {
                    f"mode{i+1}": list(S_bcoupling[i, :])
                    for i in range(0, S_bcoupling.shape[0])
                }
            ),
            "overlap": pd.DataFrame.from_dict(
                {
                    f"mode{i+1}": list(overlap_coupling[i, :])
                    for i in range(0, overlap_coupling.shape[0])
                }
            ),
            "position": pd.DataFrame.from_dict({"position": list(position)}),
        }

        return coupling_coefficients

    def plot_mode_coupling(
        self, input_mode: int, max_coupled_mode: int = None, group: int = 1
    ) -> None:
        """
        Plot mode coupling coefficients from the input_mode to the max_coupled_mode vs. position.
        Mode numbers start from 1 (fundamental mode).

        Parameters:
            input_mode: Mode of interest. Energy couples away from this mode to other modes.
            max_coupled_mode: Maximum mode number to consider; all modes below this mode will be considered as well.
            group: Group of cells to consider. First group is 0.
        """
        coupling_coefficients = self.get_mode_coupling(
            input_mode, max_coupled_mode, group
        )

        fig, axs = plt.subplots(3)
        fig.set_size_inches(8, 9)
        legend = []
        for i in range(0, len(coupling_coefficients["forward_sparam"].columns)):
            axs[0].plot(
                coupling_coefficients["position"],
                coupling_coefficients["forward_sparam"].iloc[:, i],
                marker=marker_list[i],
            )
            legend.append(coupling_coefficients["forward_sparam"].columns[i])
        axs[0].grid("on")
        axs[0].set_title("Forward Propagating S-Parameters")
        axs[0].set_ylabel("Magnitude (dB)")
        axs[0].set_xlabel("Position (m)")

        legend = []
        for i in range(0, len(coupling_coefficients["backward_sparam"].columns)):
            axs[1].plot(
                coupling_coefficients["position"],
                coupling_coefficients["backward_sparam"].iloc[:, i],
                marker=marker_list[i],
            )
            legend.append(coupling_coefficients["backward_sparam"].columns[i])
        axs[1].grid("on")
        axs[1].set_title("Backward Propagating S-Parameters")
        axs[1].set_ylabel("Magnitude (dB)")
        axs[1].set_xlabel("Position (m)")

        legend = []
        for i in range(0, len(coupling_coefficients["overlap"].columns)):
            axs[2].plot(
                coupling_coefficients["position"],
                coupling_coefficients["overlap"].iloc[:, i],
                marker=marker_list[i],
            )
            legend.append(coupling_coefficients["overlap"].columns[i])
        axs[2].grid("on")
        axs[2].set_title("Overlap Coefficients")
        axs[2].set_ylabel("Magnitude (dB)")
        axs[2].set_xlabel("Position (m)")
        fig.legend(legend, loc="center right", bbox_to_anchor=(1.15, 0.5))
        fig.suptitle(
            f"Coupling From Mode {input_mode} to Modes Up To Mode {max_coupled_mode}"
        )
        fig.tight_layout()

        fig.savefig(
            str(self.simulation_dirpath.resolve() / f"{self.component.name}_mode_coupling.png"),
            bbox_inches="tight",
        )

    def get_neff_vs_position(self, group: int = 1) -> pd.DataFrame:
        """
        Get effective index vs position along device for a given cell group

        Parameters:
            group: Group of cells to consider. First group is 0.

        Returns:
            DataFrame with neff vs position for all considered modes
            1. Effective index for different modes
            2. Position (m)
            | position | mode1 | mode2 | ...
            | float    | float | float | ...
        """

        s = self.session
        ss = self.simulation_settings

        if s.layoutmode():
            s.run()
        s.emepropagate()

        # Get starting point for number of cell and group span
        cells = [int(val[0]) for val in s.getnamed("EME", "cells")]
        num_cells = cells[group]
        group_span = round(s.getnamed("EME", "group spans")[group][0], 15)
        cell_start = 0
        for i in range(0, group):
            cell_start += cells[i]

        ## Extract neff
        neff = [[] for i in range(0, ss.num_modes)]
        for i in range(cell_start + 1, num_cells + cell_start + 1):
            data = s.getresult(f"EME::Cells::cell_{i}", "neff")
            for j in range(0, data["neff"].shape[1]):
                neff[j].append(data["neff"][0, j])

        position = np.linspace(0, group_span, num_cells + 1)[1:]
        result = {f"mode{i+1}": neff[i] for i in range(0, len(neff))}
        result["position"] = list(position)

        return pd.DataFrame.from_dict(result)

    def plot_neff_vs_position(self, group: int = 1):
        """
        Get effective index vs position along device

        Parameters:
            group: Group of cells to consider. First group is 0.
        """
        data = self.get_neff_vs_position(group)

        neff = data.filter(regex="mode")
        plt.figure()
        for i in range(0, len(neff.columns)):
            plt.plot(
                data["position"],
                np.real(neff.loc[:, neff.columns[i]]),
                label=neff.columns[i],
                marker=marker_list[i],
            )
        plt.xlabel("Position (m)")
        plt.ylabel("Effective Index")
        plt.legend(loc="upper left", bbox_to_anchor=(1.04, 1))
        plt.tight_layout()
        plt.savefig(str(self.simulation_dirpath.resolve() / f"{self.component.name}_neff_vs_position.png"))
