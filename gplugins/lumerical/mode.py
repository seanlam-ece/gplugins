from __future__ import annotations


from pathlib import Path
from typing import TYPE_CHECKING

import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gdsfactory.component import Component
from gdsfactory import logger
from gdsfactory.pdk import get_layer_stack
from gdsfactory.technology import LayerStack


from gplugins.lumerical.config import marker_list, um
from gplugins.lumerical.utils import (
    Simulation,
    draw_geometry,
    to_lbr,
)
from gplugins.lumerical.simulation_settings import LUMERICAL_MODE_SIMULATION_SETTINGS, SimulationSettingsLumericalMode
from gplugins.lumerical.convergence_settings import LUMERICAL_MODE_CONVERGENCE_SETTINGS, ConvergenceSettingsLumericalMode

if TYPE_CHECKING:
    from gdsfactory.typings import PathType


class LumericalModeSimulation(Simulation):
    def __init__(self,
        component: Component,
        layerstack: LayerStack | None = None,
        process: tuple | None = None,
        session: object | None = None,
        simulation_settings: SimulationSettingsLumericalMode = LUMERICAL_MODE_SIMULATION_SETTINGS,
        convergence_settings: ConvergenceSettingsLumericalMode = LUMERICAL_MODE_CONVERGENCE_SETTINGS,
        dirpath: PathType | None = "",
        run_port_convergence: bool = False,
        run_mesh_convergence: bool = False,
        override_convergence: bool = False,
        hide: bool = True,
        **settings,
    ):
        """
        Lumerical MODE simulation

        Set up MODE simulation from simulation settings. Optionally, run convergence sweeps
        to ensure simulations are accurate.

        Parameters:
            component: The component for which the mode simulation is to be performed.
            layerstack: PDK layerstack
            process: Process (etch, grow, implant, etc.) that affects layerstack
            session: Existing MODE session.
            simulation_settings: MODE simulation settings
            convergence_settings: MODE convergence settings
            dirpath: The directory path where simulation files are saved. Default is the current working directory.
            run_port_convergence: If True, run port convergence test and update port size. Default is False.
            run_mesh_convergence: If True, run mesh convergence test and update mesh. Default is False.
            override_convergence: If True, override convergence results and run convergence sweeps.
            hide: Hide Lumerical MODE window
            **settings: Additional simulation settings to be passed.
        """
        # Set up variables
        if isinstance(dirpath, str) and not dirpath == "":
            dirpath = Path(dirpath)
        self.dirpath = dirpath or Path(__file__).resolve().parent

        self.convergence_settings = convergence_settings or LUMERICAL_MODE_CONVERGENCE_SETTINGS
        self.component = component = gf.get_component(component)
        self.layerstack = layerstack or get_layer_stack()
        self.process = process

        # Update simulation settings with any additional information
        sim_settings = dict(simulation_settings)
        if hasattr(component.info, "simulation_settings"):
            sim_settings |= component.info.simulation_settings
            logger.info(
                f"Updating {component.name!r} sim settings {component.simulation_settings}"
            )
        for setting in settings:
            if setting not in sim_settings:
                raise ValueError(
                    f"Invalid setting {setting!r} not in ({list(sim_settings.keys())})"
                )
        sim_settings.update(**settings)
        self.simulation_settings = ss = SimulationSettingsLumericalMode(**sim_settings)

        # Initialize parent class
        super().__init__(
            component=self.component,
            layerstack=self.layerstack,
            process=self.process,
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

        # Create MODE simulation
        if not session:
            try:
                import lumapi
            except ModuleNotFoundError as e:
                print(
                    "Cannot import lumapi (Python Lumerical API). "
                    "You can add set the PYTHONPATH variable or add it with `sys.path.append()`"
                )
                raise e
            except OSError as e:
                raise e
            self.session = s = lumapi.MODE(hide=hide)
        else:
            self.session = s = session
            s.newproject()

        # Set up device geometry
        process_file_path = to_lbr(material_map=ss.material_name_to_lumerical,
                                              layerstack=self.layerstack,
                                              process=self.process,
                                              dirpath=self.simulation_dirpath)
        gdspath = self.component.write_gds(dirpath=self.simulation_dirpath)
        draw_geometry(session=self.session, gdspath=gdspath, process_file_path=process_file_path)

        # Set up simulation region
        s.addfde()
        s.set("fit materials with multi-coefficient model", 1)
        s.set("solver type", ss.injection_axis)
        s.set("x", ss.x * um)
        s.set("y", ss.y * um)
        s.set("z", ss.z * um)
        s.set("z span", ss.zspan * um)

        if ss.injection_axis == "2D X normal":
            s.set("y span", ss.yspan * um)
            s.set("y min bc", ss.ymin_boundary)
            s.set("y max bc", ss.ymax_boundary)
            s.set("define y mesh by", "maximum mesh step")
            s.set("dy", ss.wavl / ss.mesh_cells_per_wavl * um)
        elif ss.injection_axis == "2D Y normal":
            s.set("x span", ss.xspan * um)
            s.set("define x mesh by", "maximum mesh step")
            s.set("dx", ss.wavl / ss.mesh_cells_per_wavl * um)
            s.set("x min bc", ss.xmin_boundary)
            s.set("x max bc", ss.xmax_boundary)
        else:
            raise ValueError(
                f"Invalid injection axis: {ss.injection_axis}. Only 2D X normal and 2D Y normal supported."
            )

        s.set("define z mesh by", "maximum mesh step")
        s.set("dz", ss.wavl / ss.mesh_cells_per_wavl * um)
        s.set("z min bc", ss.zmin_boundary)
        s.set("z max bc", ss.zmax_boundary)

        s.set("pml layers", ss.pml_layers)

        s.setanalysis("wavelength", ss.wavl * um)
        s.setanalysis("start wavelength", ss.wavl_start * um)
        s.setanalysis("stop wavelength", ss.wavl_end * um)
        s.setanalysis("number of points", ss.wavl_pts)
        s.setanalysis("number of trial modes", ss.num_modes)
        s.setanalysis("number of test modes", ss.num_modes)
        s.setanalysis("detailed dispersion calculation", ss.include_dispersion)
        s.setanalysis("track selected mode", True)

        # Fit material models
        for layer_name in layerstack.to_dict():
            s.select("layer group")
            try:
                material_name = s.getlayer(layer_name, "pattern material")

                s.setmaterial(material_name, "wavelength min", ss.wavl_start * um)
                s.setmaterial(material_name, "wavelength max", ss.wavl_end * um)
                s.setmaterial(material_name, "tolerance", ss.material_fit_tolerance)
            except Exception:
                logger.warning(
                    f"Layer {layer_name} material cannot be found in database, skipping material fit."
                )

        # If bent waveguide analysis
        if ss.is_bent_waveguide:
            s.setanalysis("bent waveguide", True)
            s.setanalysis("bend radius", ss.bend_radius)
            s.setanalysis("bend location", ss.bend_location)
            if ss.bend_location == "user specified":
                s.setanalysis("bend location x", ss.bend_location_x)
                s.setanalysis("bend location y", ss.bend_location_y)
                s.setanalysis("bend location z", ss.bend_location_z)

        s.save(str(self.simulation_dirpath.resolve() / f"{self.component.name}.lms"))

        # Run convergence testing if no convergence results are available or user wants to override convergence results
        # or if setup has changed
        if (
                not self.convergence_results.available()
                or override_convergence
                or not self.convergence_is_fresh()
        ):
            if run_mesh_convergence:
                self.convergence_results.mesh_convergence_data = self.update_mesh_convergence(
                    plot=not hide,
                    verbose=not hide,
                )

            if run_port_convergence:
                self.convergence_results.field_intensity_convergence_data = self.update_field_intensity_threshold(
                    plot=not hide,
                    verbose=not hide,
                )
                self.update_port_convergence(verbose=not hide)



            if (run_mesh_convergence and run_port_convergence):
                # Save setup and results for convergence
                self.save_convergence_results()
                if not hide:
                    logger.info("Saved convergence results.")

    def update_mesh_convergence(self, plot: bool = False, verbose: bool = False) -> pd.DataFrame:
        """
        Update mesh and simulation setup by performing convergence testing on mesh (mesh cells per wavelength)

        Parameters:
            verbose: Log debug messages and convergence status

        Returns:
            Mesh convergence results
            | mesh_cells_per_wavelength | neff_r | neff_i | ng_r  | ng_i  | pol   |
            | int                       | float  | float  | float | float | float |
        """
        s = self.session
        ss = self.simulation_settings
        cs = self.convergence_settings

        mesh_step = cs.mesh_cell_step
        limit = cs.mesh_stable_limit
        dneff = cs.neff_diff
        dng = cs.ng_diff
        dpol = cs.pol_diff

        # Get initial mesh cell settings
        s.switchtolayout()
        s.select("FDE")
        injection_direction = s.get("solver type")
        wavl = s.get("wavelength")
        if injection_direction == "2D X normal":
            mesh_cells_per_wavl = wavl / s.get("dy")
        elif injection_direction == "2D Y normal":
            mesh_cells_per_wavl = wavl / s.get("dx")
        mesh_cells_per_wavl = max([mesh_cells_per_wavl, wavl / s.get("dz")])

        neff_r = []
        neff_i = []
        ng_r = []
        ng_i = []
        pol = []

        mesh_cells_ratio = []

        # Sweep number of mesh cells per wavelength until neff, ng, and pol fraction converge
        converged = False
        while not converged:
            s.switchtolayout()
            s.select("FDE")
            if injection_direction == "2D X normal":
                s.set("dy", wavl / mesh_cells_per_wavl)
            elif injection_direction == "2D Y normal":
                s.set("dx", wavl / mesh_cells_per_wavl)
            s.set("dz", wavl / mesh_cells_per_wavl)

            s.mesh()
            s.findmodes()
            s.selectmode(ss.target_mode)

            neff_r.append(
                np.real(s.getresult(f"FDE::data::mode{ss.target_mode}", "neff"))[0][0]
            )
            neff_i.append(
                np.imag(s.getresult(f"FDE::data::mode{ss.target_mode}", "neff"))[0][0]
            )
            ng_r.append(
                np.real(s.getresult(f"FDE::data::mode{ss.target_mode}", "ng"))[0][0]
            )
            ng_i.append(
                np.imag(s.getresult(f"FDE::data::mode{ss.target_mode}", "ng"))[0][0]
            )
            pol.append(
                s.getresult(f"FDE::data::mode{ss.target_mode}", "TE polarization fraction")
            )

            mesh_cells_ratio.append(mesh_cells_per_wavl)

            # Check convergence
            if (
                    len(neff_r) >= limit
                    and len(neff_i) >= limit
                    and len(ng_r) >= limit
                    and len(ng_i) >= limit
                    and len(pol) >= limit
            ):
                converged = (
                        all([abs(neff_r[-1] - neff_r[-i]) < dneff for i in range(2, limit + 1)])
                        and all(
                    [abs(neff_i[-1] - neff_i[-i]) < dneff for i in range(2, limit + 1)]
                )
                        and all([abs(ng_r[-1] - ng_r[-i]) < dng for i in range(2, limit + 1)])
                        and all([abs(ng_i[-1] - ng_i[-i]) < dng for i in range(2, limit + 1)])
                        and all([abs(pol[-1] - pol[-i]) < dpol for i in range(2, limit + 1)])
                )

            mesh_cells_per_wavl += mesh_step

            if verbose:
                logger.info(f"mesh cells per {wavl}: {mesh_cells_per_wavl}.")

        # Once converged, set mesh cells
        s.switchtolayout()
        s.select("FDE")
        if injection_direction == "2D X normal":
            s.set("dy", wavl / mesh_cells_per_wavl)
        elif injection_direction == "2D Y normal":
            s.set("dx", wavl / mesh_cells_per_wavl)
        s.set("dz", wavl / mesh_cells_per_wavl)
        ss.mesh_cells_per_wavl = mesh_cells_per_wavl

        s.save()

        if plot:
            plt.plot(mesh_cells_ratio, neff_r, label="Real(neff)")
            plt.plot(mesh_cells_ratio, neff_i, label="Imag(neff)")
            plt.plot(mesh_cells_ratio, ng_r, label="Real(ng)")
            plt.plot(mesh_cells_ratio, ng_i, label="Imag(ng)")
            plt.plot(mesh_cells_ratio, pol, label="TE Polarization Fraction")
            plt.title("Mesh Cell Convergence")
            plt.xlabel("Mesh Cells Per Wavelength")
            plt.ylabel("Magnitude")
            plt.grid("on")
            plt.legend()
            plt.savefig(str(self.simulation_dirpath.resolve() / "mesh_convergence.png"))

        convergence_results = pd.DataFrame(
            {
                "mesh_cells_per_wavelength": mesh_cells_ratio,
                "neff_r": neff_r,
                "neff_i": neff_i,
                "ng_r": ng_r,
                "ng_i": ng_i,
                "pol": pol,
            }
        )
        return convergence_results

    def update_port_convergence(self, verbose: bool = False):
        """
        Update port size to ensure E-field intensities decay to set threshold and updates simulation setup to
        reflect changes.

        Parameters:
            verbose: Log debug messages and convergence status
        """
        s = self.session
        ss = self.simulation_settings

        threshold = ss.efield_intensity_threshold

        # Set simulation region arbitrarily large to then calculate where E-field intensities decay to threshold
        s.switchtolayout()
        s.select("FDE")
        injection_direction = s.get("solver type")
        if injection_direction == "2D X normal":
            s.set("y span", s.get("y span") * 2)
        elif injection_direction == "2D Y normal":
            s.set("x span", s.get("x span") * 2)
        s.set("z span", s.get("z span") * 2)

        # Check port size convergence. Check that E field intensities decay to threshold at edges
        converged = False
        while not converged:
            s.mesh()
            s.findmodes()
            s.selectmode(ss.target_mode)

            s.eval(
                f'mode_profiles=getresult("FDE::data::mode{ss.target_mode}","E");'
                + f"E=mode_profiles.E; x=mode_profiles.x; y=mode_profiles.y; z=mode_profiles.z;"
            )
            E = s.getv("E")
            x = s.getv("x")
            y = s.getv("y")
            z = s.getv("z")

            s.switchtolayout()

            # To get E field intensity, need to find port orientation
            # The E field intensity data depends on injection axis
            if injection_direction == "2D X normal":
                Efield_xyz = np.array(E[0, :, :, 0, :])
            elif injection_direction == "2D Y normal":
                Efield_xyz = np.array(E[:, 0, :, 0, :])

            Efield_intensity = np.empty([Efield_xyz.shape[0], Efield_xyz.shape[1]])
            for a in range(0, Efield_xyz.shape[0]):
                for b in range(0, Efield_xyz.shape[1]):
                    Efield_intensity[a, b] = (
                            abs(Efield_xyz[a, b, 0]) ** 2
                            + abs(Efield_xyz[a, b, 1]) ** 2
                            + abs(Efield_xyz[a, b, 2]) ** 2
                    )

            # Get max E field intensity along x/y axis
            Efield_intensity_xy = np.empty([Efield_intensity.shape[0]])
            for a in range(0, Efield_intensity.shape[0]):
                Efield_intensity_xy[a] = max(Efield_intensity[a, :])

            # Get max E field intensity along z axis
            Efield_intensity_z = np.empty([Efield_intensity.shape[1]])
            for b in range(0, Efield_intensity.shape[1]):
                Efield_intensity_z[b] = max(Efield_intensity[:, b])

            # Get initial z min and z max for expansion reference
            # Get initial z span;  this will be used to expand ports
            s.select("FDE")
            port_z_min = s.get("z min")
            port_z_max = s.get("z max")
            port_z_span = s.get("z span")

            # If all E field intensities > threshold, expand z span of port by initial z span
            # Else, set z min and z max to locations where E field intensities decay below threshold
            indexes = np.argwhere(Efield_intensity_z > threshold)
            if len(indexes) == 0:
                min_index = 0
                max_index = len(Efield_intensity_z) - 1
            else:
                min_index, max_index = int(np.min(indexes)), int(np.max(indexes))

            if min_index == 0:
                s.set("z min", port_z_min - port_z_span / 2)
                converged_zmin = False
            else:
                s.set("z min", z[min_index - 1])
                converged_zmin = True

            if max_index == (len(Efield_intensity_z) - 1):
                s.set("z max", port_z_max + port_z_span / 2)
                converged_zmax = False
            else:
                s.set("z max", z[max_index + 1])
                converged_zmax = True

            if verbose:
                logger.info(
                    f"mode {ss.target_mode} field decays at z: {z[max_index]}, {z[min_index]} microns"
                )

            # Get initial x/y min and x/y max for expansion reference
            # Get initial x/y span;  this will be used to expand ports
            if injection_direction == "2D X normal":
                port_xy_min = s.get("y min")
                port_xy_max = s.get("y max")
                port_xy_span = s.get("y span")

                # If all E field intensities > threshold, expand x/y span of port by initial x/y span
                # Else, set x/y min and x/y max to locations where E field intensities decay below threshold
                indexes = np.argwhere(Efield_intensity_xy > threshold)
                if len(indexes) == 0:
                    min_index = 0
                    max_index = len(Efield_intensity_xy) - 1
                else:
                    min_index, max_index = int(np.min(indexes)), int(np.max(indexes))

                if min_index == 0:
                    s.set("y min", port_xy_min - port_xy_span / 2)
                    converged_ymin = False
                else:
                    s.set("y min", y[min_index - 1])
                    converged_ymin = True

                if max_index == (len(Efield_intensity_xy) - 1):
                    s.set("y max", port_xy_max + port_xy_span / 2)
                    converged_ymax = False
                else:
                    s.set("y max", y[max_index + 1])
                    converged_ymax = True

                if verbose:
                    logger.info(
                        f"mode {ss.target_mode} field decays at y: {y[max_index]}, {y[min_index]} microns"
                    )

                converged = (
                        converged_ymax & converged_ymin & converged_zmax & converged_zmin
                )

            elif injection_direction == "2D Y normal":
                port_xy_min = s.get("x min")
                port_xy_max = s.get("x max")
                port_xy_span = s.get("x span")

                # If all E field intensities > threshold, expand x/y span of port by initial x/y span
                # Else, set x/y min and x/y max to locations where E field intensities decay below threshold
                indexes = np.argwhere(Efield_intensity_xy > threshold)
                if len(indexes) == 0:
                    min_index = 0
                    max_index = len(Efield_intensity_xy) - 1
                else:
                    min_index, max_index = int(min(indexes)), int(max(indexes))

                if min_index == 0:
                    s.set("x min", port_xy_min - port_xy_span / 2)
                    converged_xmin = False
                else:
                    s.set("x min", x[min_index - 1])
                    converged_xmin = True

                if max_index == (len(Efield_intensity_xy) - 1):
                    s.set("x max", port_xy_max + port_xy_span / 2)
                    converged_xmax = False
                else:
                    s.set("x max", x[max_index + 1])
                    converged_xmax = True

                if verbose:
                    logger.info(
                        f"mode {ss.target_mode} field decays at x: {x[max_index]}, {x[min_index]} microns"
                    )

                converged = (
                        converged_xmax & converged_xmin & converged_zmax & converged_zmin
                )

        # Once converged, record margin distance between device and simulation region into simulation setup
        s.switchtolayout()
        s.select("FDE")
        ss.xspan = s.get("x span") / um
        ss.yspan = s.get("y span") / um
        ss.zspan = s.get("z span") / um

        s.save()

    def update_field_intensity_threshold(
        self,
        plot: bool = False,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Update port field intensity threshold based on sweep of field intensity and neff, ng, polarization.

        Parameters:
            plot: Plot sparam convergence results and save png

        Returns:
            Convergence data in pd.DataFrame.
            | thresholds | neff_r  | neff_i  | ng_r  | ng_i  | pol   |
            | float      | float   | float   | float | float | float |
        """
        s = self.session
        ss = self.simulation_settings
        cs = self.convergence_settings

        limit = cs.field_stable_limit
        dneff = cs.neff_diff
        dng = cs.ng_diff
        dpol = cs.pol_diff

        neff_r = []
        neff_i = []
        ng_r = []
        ng_i = []
        pol = []

        efield_intensity_threshold = 1e-1
        thresholds = []
        converged = False
        while not converged:
            ss.efield_intensity_threshold = efield_intensity_threshold
            thresholds.append(efield_intensity_threshold)
            self.update_port_convergence(verbose=plot)

            s.mesh()
            s.findmodes()
            s.selectmode(ss.target_mode)

            neff_r.append(
                np.real(s.getresult(f"FDE::data::mode{ss.target_mode}", "neff"))[0][0]
            )
            neff_i.append(
                np.imag(s.getresult(f"FDE::data::mode{ss.target_mode}", "neff"))[0][0]
            )
            ng_r.append(
                np.real(s.getresult(f"FDE::data::mode{ss.target_mode}", "ng"))[0][0]
            )
            ng_i.append(
                np.imag(s.getresult(f"FDE::data::mode{ss.target_mode}", "ng"))[0][0]
            )
            pol.append(
                s.getresult(f"FDE::data::mode{ss.target_mode}", "TE polarization fraction")
            )

            # Check convergence
            if (
                    len(neff_r) >= limit
                    and len(neff_i) >= limit
                    and len(ng_r) >= limit
                    and len(ng_i) >= limit
                    and len(pol) >= limit
            ):
                converged = (
                        all([abs(neff_r[-1] - neff_r[-i]) < dneff for i in range(2, limit + 1)])
                        and all(
                    [abs(neff_i[-1] - neff_i[-i]) < dneff for i in range(2, limit + 1)]
                )
                        and all([abs(ng_r[-1] - ng_r[-i]) < dng for i in range(2, limit + 1)])
                        and all([abs(ng_i[-1] - ng_i[-i]) < dng for i in range(2, limit + 1)])
                        and all([abs(pol[-1] - pol[-i]) < dpol for i in range(2, limit + 1)])
                )

            if verbose:
                logger.info(f"E-Field Intensity Threshold: {efield_intensity_threshold}.")

            efield_intensity_threshold = efield_intensity_threshold / 10

        s.save()

        if plot:
            plt.plot(thresholds, neff_r, label="Real(neff)", marker=marker_list[0])
            plt.plot(thresholds, neff_i, label="Imag(neff)", marker=marker_list[1])
            plt.plot(thresholds, ng_r, label="Real(ng)", marker=marker_list[2])
            plt.plot(thresholds, ng_i, label="Imag(ng)", marker=marker_list[3])
            plt.plot(thresholds, pol, label="TE Polarization Fraction", marker=marker_list[4])
            plt.title("E-Field Intensity Threshold Convergence")
            plt.xscale("log")
            plt.xlabel("E-Field Intensity (V/m)^2")
            plt.ylabel("Magnitude")
            plt.grid("on")
            plt.legend()
            plt.savefig(str(self.simulation_dirpath.resolve() / "efield_intensity_threshold_convergence.png"))

        convergence_results = pd.DataFrame(
            {
                "thresholds": thresholds,
                "neff_r": neff_r,
                "neff_i": neff_i,
                "ng_r": ng_r,
                "ng_i": ng_i,
                "pol": pol,
            }
        )
        return convergence_results

    def plot_index(self) -> None:
        """
        Plot cross section index
        """
        m = self.session
        m.mesh()

        injection_axis = m.getnamed("FDE", "solver type")
        if injection_axis == "2D X normal":
            index = np.transpose(m.getdata("FDE::data::material", "index_x")[0, :, :, 0])
            xymax = m.getnamed("FDE", "y max") / um
            xymin = m.getnamed("FDE", "y min") / um
            zmax = m.getnamed("FDE", "z max") / um
            zmin = m.getnamed("FDE", "z min") / um
        else:
            raise ValueError(f"{injection_axis} unsupported")

        cmap = "jet"
        origin = "lower"
        fig1 = plt.figure(figsize=(8, 6))
        ax1 = fig1.add_subplot(111)

        im = ax1.imshow(
            index.real,
            cmap=cmap,
            origin=origin,
            aspect="auto",
            extent=[xymin, xymax, zmin, zmax],
        )
        ax1.set_title("Index")
        ax1.set_xlabel("X/Y (um)")
        ax1.set_ylabel("Z (um)")
        fig1.colorbar(im)
        fig1.savefig(str(self.simulation_dirpath.resolve() / "index.png"))
        plt.show()

    def get_neff_vs_wavelength(self) -> pd.DataFrame:
        """
        Retrieve effective index (neff) versus wavelength data.

        Returns:
            A dataframe containing wavelength (in micrometers) and corresponding effective indices (neff).
            | wavl  | neff    |
            | float | complex |
        """
        ss = self.simulation_settings

        # Run simulation
        m = self.session
        m.run()
        m.mesh()
        m.findmodes()
        m.selectmode(ss.target_mode)
        m.frequencysweep()
        neff = [val[0] for val in m.getdata("FDE::data::frequencysweep", "neff")]
        wavl = [val[0] for val in m.c() / m.getdata("FDE::data::frequencysweep", "f") / um]
        neff_vs_wavl = pd.DataFrame.from_dict({"wavl": wavl, "neff": neff})

        return neff_vs_wavl

    def plot_neff_vs_wavelength(self, neff_vs_wavl: pd.DataFrame = pd.DataFrame()) -> None:
        """
        Plot effective index vs. wavelength and saves plot

        Parameters:
            neff_vs_wavl: Effective index vs wavelength data
        """
        if neff_vs_wavl.empty:
            neff_vs_wavl = self.get_neff_vs_wavelength()

        plt.plot(neff_vs_wavl["wavl"], neff_vs_wavl["neff"])
        plt.title("Effective Index")
        plt.xlabel("Wavelength (um)")
        plt.ylabel("Effective Index")
        plt.grid("on")
        plt.savefig(str(self.simulation_dirpath.resolve() / "neff_vs_wavelength.png"))
        plt.show()

    def get_ng_vs_wavelength(self) -> pd.DataFrame:
        """
        Retrieve group index (ng) versus wavelength data.

        Returns:
            A dataframe containing wavelength (in micrometers) and corresponding group indices (ng).
            | wavl  | ng      |
            | float | complex |
        """
        ss = self.simulation_settings

        # Run simulation
        m = self.session
        m.run()
        m.mesh()
        m.findmodes()
        m.selectmode(ss.target_mode)
        m.frequencysweep()
        ng = [val[0] for val in m.c() / m.getdata("FDE::data::frequencysweep", "vg")]
        wavl = [val[0] for val in m.c() / m.getdata("FDE::data::frequencysweep", "f") / um]
        ng_vs_wavl = pd.DataFrame.from_dict({"wavl": wavl, "ng": ng})

        return ng_vs_wavl

    def plot_ng_vs_wavelength(self, ng_vs_wavl: pd.DataFrame = pd.DataFrame()):
        """
        Plot group index vs. wavelength and saves plot

        Parameters:
            ng_vs_wavl: Group index vs wavelength data
        """
        if ng_vs_wavl.empty:
            ng_vs_wavl = self.get_ng_vs_wavelength()

        plt.plot(ng_vs_wavl["wavl"], ng_vs_wavl["ng"])
        plt.title("Group Index")
        plt.xlabel("Wavelength (um)")
        plt.ylabel("Group Index")
        plt.grid("on")
        plt.savefig(str(self.simulation_dirpath.resolve() / "ng_vs_wavl.png"))
        plt.show()

    def get_dispersion_vs_wavelength(self) -> pd.DataFrame:
        """
        Retrieve dispersion (s/m/m) versus wavelength data.

        Returns:
             A dataframe with wavelength (in micrometers) and corresponding dispersion (s/m/m).
             | wavl  | D      |
            | float  | float |

        """
        ss = self.simulation_settings
        # Run simulation
        m = self.session
        m.run()
        m.mesh()
        m.findmodes()
        m.selectmode(ss.target_mode)
        m.frequencysweep()
        D = [val[0] for val in m.getdata("FDE::data::frequencysweep", "D")]
        wavl = [val[0] for val in m.c() / m.getdata("FDE::data::frequencysweep", "f") / um]
        D_vs_wavl = pd.DataFrame.from_dict({"wavl": wavl, "D": D})

        return D_vs_wavl

    def plot_dispersion_vs_wavelength(self, D_vs_wavl: pd.DataFrame = pd.DataFrame()):
        """
        Plot dispersion vs. wavelength and saves plot

        Parameters:
            D_vs_wavl: Dispersion vs wavelength data
        """
        if D_vs_wavl.empty:
            D_vs_wavl = self.get_dispersion_vs_wavelength()

        plt.plot(D_vs_wavl["wavl"], D_vs_wavl["D"])
        plt.title("Dispersion")
        plt.xlabel("Wavelength (um)")
        plt.ylabel("Dispersion (s/m/m)")
        plt.grid("on")
        plt.savefig(str(self.simulation_dirpath.resolve() / "dispersion_vs_wavl.png"))
        plt.show()

