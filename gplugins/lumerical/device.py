from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import gdsfactory as gf
import numpy as np
from gdsfactory.component import Component
from gdsfactory import logger
from gdsfactory.pdk import get_layer_stack
from gdsfactory.technology import LayerStack

from gplugins.lumerical.config import cm, um, dbu, OPACITY, MATERIAL_COLORS
from gplugins.lumerical.convergence_settings import (
    LUMERICAL_CHARGE_CONVERGENCE_SETTINGS,
    ConvergenceSettingsLumericalCharge,
)
from gplugins.lumerical.simulation_settings import (
    LUMERICAL_CHARGE_SIMULATION_SETTINGS,
    SimulationSettingsLumericalCharge,
)
from gplugins.lumerical.utils import Simulation, draw_geometry, to_lbr

if TYPE_CHECKING:
    from gdsfactory.typings import PathType


class LumericalChargeSimulation(Simulation):
    """
    Lumerical CHARGE simulation

    Set up CHARGE simulation based on component geometry and simulation settings. Optionally, run convergence.

    Attributes:
        component: Component geometry to simulate
        layerstack: PDK layerstack
        process: Process (etch, grow, implant, etc.) that affects layerstack
        session: Lumerical session
        simulation_settings: CHARGE simulation settings
        convergence_settings: CHARGE convergence settings
        dirpath: Directory where simulation files are saved

    """

    def __init__(
        self,
        component: Component,
        layerstack: LayerStack | None = None,
        process: tuple | None = None,
        session: object | None = None,
        simulation_settings: SimulationSettingsLumericalCharge = LUMERICAL_CHARGE_SIMULATION_SETTINGS,
        convergence_settings: ConvergenceSettingsLumericalCharge = LUMERICAL_CHARGE_CONVERGENCE_SETTINGS,
        boundary_settings: dict[str, dict] | None = None,
        dirpath: PathType | None = "",
        hide: bool = True,
        **settings,
    ):
        if isinstance(dirpath, str) and not dirpath == "":
            dirpath = Path(dirpath)
        self.dirpath = dirpath or Path(__file__).resolve().parent

        self.convergence_settings = (
            convergence_settings or LUMERICAL_CHARGE_CONVERGENCE_SETTINGS
        )
        self.component = gf.get_component(component)
        self.layerstack = layerstack or get_layer_stack()
        self.process = process

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
        self.simulation_settings = SimulationSettingsLumericalCharge(**sim_settings)

        super().__init__(
            component=self.component,
            layerstack=self.layerstack,
            process=self.process,
            simulation_settings=self.simulation_settings,
            convergence_settings=self.convergence_settings,
            dirpath=self.dirpath,
        )

        ss = self.simulation_settings

        ### Get new CHARGE session
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
        self.session = s = session or lumapi.DEVICE(hide=hide)
        s.newproject()
        s.deleteall()

        # Add materials before drawing geometry. This is necessary since Layer Builder references these materials
        self.add_charge_materials()

        # Draw geometry
        combined_material_map = ss.optical_material_name_to_lumerical.copy()
        combined_material_map.update(ss.ele_therm_material_name_to_lumerical)
        process_file_path = to_lbr(
            material_map=combined_material_map,
            layerstack=self.layerstack,
            process=self.process,
            dirpath=self.simulation_dirpath.resolve(),
            use_pdk_material_names=True,
        )
        gdspath = self.component.write_gds(gdsdir=self.simulation_dirpath.resolve())
        draw_geometry(session=s, gdspath=gdspath, process_file_path=process_file_path)

        # Add and configure simulation region
        self.add_simulation_region()

        # Add boundary conditions
        self.add_boundary_conditions(boundary_settings=boundary_settings)

        self.add_charge_monitor()

        s.partitionvolume()

        s.save(str(self.simulation_dirpath.resolve() / f"{self.component.name}.ldev"))

    def add_charge_materials(
        self,
        optical_layer_map: dict[str, str] = None,
        ele_therm_layer_map: dict[str, str] = None,
        material_fit_tolerance: float | None = None,
    ):
        """
        Add materials to simulation.

        Materials need to be added prior to assigning these materials to structures.

        Parameters:
            optical_layer_map: Map of optical materials from PDK materials to Lumerical materials
            ele_therm_layer_map: Map of electrical and thermal materials from PDK materials to Lumerical materials
            material_fit_tolerance: Optical material fit tolerance
        """

        s = self.session
        optical_layer_map = (
            optical_layer_map
            or self.simulation_settings.optical_material_name_to_lumerical
        )
        ele_therm_layer_map = (
            ele_therm_layer_map
            or self.simulation_settings.ele_therm_material_name_to_lumerical
        )
        material_fit_tolerance = (
            material_fit_tolerance or self.simulation_settings.material_fit_tolerance
        )

        # Remove any initial materials
        s.groupscope("::model::materials")
        s.deleteall()
        s.groupscope("::model")

        s.addmodelmaterial()
        ele_therm_materials = s.addmaterialproperties("CT").split("\n")
        opt_materials = s.addmaterialproperties("EM").split("\n")
        s.delete("New Material")

        # Add materials only supported by Lumerical
        i = 0
        for name, material in optical_layer_map.items():
            if material not in opt_materials:
                logger.warning(
                    f"{material} is not a supported optical material in Lumerical and will be skipped."
                )
                continue
            else:
                s.addmodelmaterial()
                s.set("name", name)
                s.set("color", MATERIAL_COLORS[i])
                s.addmaterialproperties("EM", material)
                s.set("tolerance", material_fit_tolerance)
                i += 1

        for name, material in ele_therm_layer_map.items():
            if material not in ele_therm_materials:
                logger.warning(
                    f"{material} is not a supported electrical or thermal material in Lumerical and will be skipped."
                )
                continue
            else:
                if not s.materialexists(name):
                    s.addmodelmaterial()
                    s.set("name", name)
                    s.set("color", MATERIAL_COLORS[i])
                    i += 1
                else:
                    s.select(f"materials::{name}")
                s.addmaterialproperties("CT", material)

    def add_simulation_region(
        self,
        simulation_settings: SimulationSettingsLumericalCharge | None = None,
        convergence_settings: ConvergenceSettingsLumericalCharge | None = None,
    ):
        """
        Set simulation region geometry and boundaries

        Parameters:
            simulation_settings: CHARGE simulation settings
        """
        s = self.session
        ss = simulation_settings or self.simulation_settings
        cs = convergence_settings or self.convergence_settings
        self.simulation_settings = ss
        self.convergence_settings = cs

        s.select("simulation region")
        s.set("dimension", ss.dimension)
        s.set("x min boundary", ss.xmin_boundary)
        s.set("x max boundary", ss.xmax_boundary)
        s.set("y min boundary", ss.ymin_boundary)
        s.set("y max boundary", ss.ymax_boundary)
        s.set("z min boundary", ss.zmin_boundary)
        s.set("z max boundary", ss.zmax_boundary)
        s.set("x", ss.x * um)
        s.set("y", ss.y * um)
        s.set("z", ss.z * um)
        s.set("x span", ss.xspan * um)
        s.set("y span", ss.yspan * um)
        s.set("z span", ss.zspan * um)

        # Delete existing solver to create new one
        s.select("CHARGE")
        s.delete()

        s.addchargesolver()
        # Set general settings
        s.set("norm length", ss.norm_length * um)
        s.set("solver mode", ss.solver_mode)
        s.set("temperature dependence", ss.temperature_dependence)
        s.set("simulation temperature", ss.simulation_temperature)
        # Set mesh
        s.set("min edge length", ss.min_edge_length * um)
        s.set("max edge length", ss.max_edge_length * um)
        s.set("max refine steps", ss.max_refine_steps)
        # Set transient simulation settings
        s.set("transient min time step", ss.min_time_step)
        s.set("transient max time step", ss.max_time_step)
        # Set AC simulation settings
        s.set("perturbation amplitude", ss.vac_amplitude)
        s.set("frequency spacing", ss.frequency_spacing)
        # Set frequency partitioning settings for AC analysis
        if ss.frequency_spacing == "single":
            s.set("frequency", ss.frequency)
        elif ss.frequency_spacing == "linear":
            s.set("start frequency", ss.start_frequency)
            s.set("stop frequency", ss.stop_frequency)
            s.set("num frequency points", ss.num_frequency_pts)
        elif ss.frequency_spacing == "log":
            s.set("log start frequency", ss.start_frequency)
            s.set("log stop frequency", ss.stop_frequency)
            s.set("num frequency points per dec", ss.num_frequency_pts)

        # Set convergence settings
        s.set("solver type", cs.solver_type)
        s.set("dc update mode", cs.dc_update_mode)
        s.set("global iteration limit", cs.global_iteration_limit)
        s.set("gradient mixing", cs.gradient_mixing)
        s.set("convergence criteria", cs.convergence_criteria)
        s.set("update abs tol", cs.update_abs_tol)
        s.set("update rel tol", cs.update_rel_tol)
        s.set("residual abs tol", cs.residual_abs_tol)

    def add_boundary_conditions(self, boundary_settings: dict[str, dict] | None = None):
        """
        Add electrical boundary conditions and surface recombination boundary conditions to simulation.

        Electrical boundary conditions are created for polygons on the metal_layer (from simulation_settings) that are
        inside the simulation region. Surface recombination boundary conditions are created for material:material
        contact locations between the metal_layer and dopant_layer (from simulation_settings).

        Boundaries are named b0, b1, b2, etc. and are labeled from left to right of the simulation region.

        Parameters:
            boundary_settings: Mapping between boundary condition name to its electrical boundary condition settings.
                                Ex. {
                                        "b0":
                                        {
                                            "bc mode": "steady state",
                                            "sweep type": "single",
                                            "voltage": 1.1,
                                            "force ohmic": True
                                        },
                                        "b1":
                                        {
                                            "bc mode": "steady state",
                                            "sweep type": "single",
                                            "voltage": 0.1,
                                            "force ohmic": True
                                        }
                                    }

        """
        ss = self.simulation_settings
        s = self.session
        c = self.component

        # To set the boundary conditions, we need to know the location of the metal contacts
        # This is given by the metal_layer in simulation settings

        # But, this will only give a list of polygons and coordinates
        # We will also need the simulation region orientation and location to narrow down where the contacts are

        # Get metal layer polygons. Each polygon is a metal boundary condition if it is inside the simulation region.
        layer_spec = (
            (self.layerstack.layers[ss.metal_layer].layer.layer.layer, self.layerstack.layers[ss.metal_layer].layer.layer.datatype)
            if isinstance(ss.metal_layer, str)
            else ss.metal_layer
        )
        polygons = c.get_polygons(by="tuple", layers=[layer_spec])

        # Get simulation region orientation and bounds
        # Get list of coords that cross the simulation plane

        zmin = self.layerstack.get_layer_to_zmin()[layer_spec]
        thickness = self.layerstack.get_layer_to_thickness()[layer_spec]
        z = (
            np.mean([zmin, ss.z + ss.zspan / 2])
            if zmin + thickness / 2 > ss.z + ss.zspan / 2
            else zmin + thickness / 2
        )
        bound_coords = []
        polygons = polygons[layer_spec]
        if ss.dimension == "2D X-Normal":
            for i in range(0, len(polygons)):
                poly_coords = np.array([[p.x / dbu, p.y / dbu] for p in polygons[i].each_point_hull()])
                coords = []
                for j in range(0, len(poly_coords) - 1):
                    # If x coord crosses the simulation plane, continue to check whether the points are in the
                    # simulation region
                    if (
                        poly_coords[j, 0] <= ss.x <= poly_coords[j + 1, 0]
                        or poly_coords[j, 0] >= ss.x >= poly_coords[j + 1, 0]
                    ):
                        x = ss.x
                        # Sort the x coords such that lower coord is first, needed for numpy's linear interpolation.
                        xp = (
                            [poly_coords[j, 0], poly_coords[j + 1, 0]]
                            if poly_coords[j, 0] < poly_coords[j + 1, 0]
                            else [poly_coords[j + 1, 0], poly_coords[j, 0]]
                        )
                        yp = (
                            [poly_coords[j, 1], poly_coords[j + 1, 1]]
                            if poly_coords[j, 0] < poly_coords[j + 1, 0]
                            else [poly_coords[j + 1, 1], poly_coords[j, 1]]
                        )
                        y = np.interp(ss.x, xp, yp)

                        # If y coord is in the simulation span, add coords
                        if ss.y - ss.yspan / 2 < y < ss.y + ss.yspan / 2:
                            coords.append(np.array([x, y]))
                    if (
                            (j == 0 and poly_coords[0, 0] <= ss.x <= poly_coords[-1, 0])
                            or (j == 0 and poly_coords[0, 0] >= ss.x >= poly_coords[-1, 0])
                    ):
                        x = ss.x
                        # Sort the x coords such that lower coord is first, needed for numpy's linear interpolation.
                        xp = (
                            [poly_coords[0, 0], poly_coords[-1, 0]]
                            if poly_coords[0, 0] < poly_coords[-1, 0]
                            else [poly_coords[-1, 0], poly_coords[0, 0]]
                        )
                        yp = (
                            [poly_coords[0, 1], poly_coords[-1, 1]]
                            if poly_coords[0, 0] < poly_coords[-1, 0]
                            else [poly_coords[-1, 1], poly_coords[0, 1]]
                        )
                        y = np.interp(ss.x, xp, yp)

                        # If y coord is in the simulation span, add coords
                        if ss.y - ss.yspan / 2 < y < ss.y + ss.yspan / 2:
                            coords.append(np.array([x, y]))

                if coords:
                    coords = np.array(coords)
                    bound_coords.append([ss.x * um, np.mean(coords[:, 1]) * um, z * um])

        elif ss.dimension == "2D Y-Normal":
            for i in range(0, len(polygons)):
                poly_coords = np.array([[p.x / dbu, p.y / dbu] for p in polygons[i].each_point_hull()])
                coords = []
                for j in range(0, len(poly_coords) - 1):
                    # If y coord crosses the simulation plane, continue to check whether the points are in the
                    # simulation region
                    if (
                        poly_coords[j, 1] <= ss.y <= poly_coords[j + 1, 1]
                        or poly_coords[j, 1] >= ss.y >= poly_coords[j + 1, 1]
                    ):
                        y = ss.y
                        # Sort the y coords such that lower coord is first, needed for numpy's linear interpolation.
                        yp = (
                            [poly_coords[j, 1], poly_coords[j + 1, 1]]
                            if poly_coords[j, 1] < poly_coords[j + 1, 1]
                            else [poly_coords[j + 1, 1], poly_coords[j, 1]]
                        )
                        xp = (
                            [poly_coords[j, 0], poly_coords[j + 1, 0]]
                            if poly_coords[j, 1] < poly_coords[j + 1, 1]
                            else [poly_coords[j + 1, 0], poly_coords[j, 0]]
                        )
                        x = np.interp(ss.y, yp, xp)

                        # If y coord is in the simulation span, add coords
                        if ss.x - ss.xspan / 2 < x < ss.x + ss.xspan / 2:
                            coords.append(np.array([x, y]))
                    if (
                            (j == 0 and poly_coords[0, 1] <= ss.y <= poly_coords[-1, 1])
                            or (j == 0 and poly_coords[0, 1] >= ss.y >= poly_coords[-1, 1])
                    ):
                        y = ss.y
                        # Sort the y coords such that lower coord is first, needed for numpy's linear interpolation.
                        yp = (
                            [poly_coords[0, 1], poly_coords[-1, 1]]
                            if poly_coords[0, 1] < poly_coords[-1, 1]
                            else [poly_coords[-1, 1], poly_coords[0, 1]]
                        )
                        xp = (
                            [poly_coords[0, 0], poly_coords[-1, 0]]
                            if poly_coords[0, 1] < poly_coords[-1, 1]
                            else [poly_coords[-1, 0], poly_coords[0, 0]]
                        )
                        x = np.interp(ss.y, yp, xp)

                        # If x coord is in the simulation span, add coords
                        if ss.x - ss.xspan / 2 < x < ss.x + ss.xspan / 2:
                            coords.append(np.array([x, y]))

                if coords:
                    coords = np.array(coords)
                    bound_coords.append([np.mean(coords[:, 0]) * um, ss.y * um, z * um])

        # Delete all existing boundary conditions
        s.groupscope("::model::CHARGE::boundary conditions")
        s.deleteall()
        s.groupscope("::model")

        # Sort boundary coordinates to ensure we are naming these boundaries from left to right
        bound_coords.sort()

        # Create and set coordinates of boundaries
        self.boundary_condition_settings = {}
        for i in range(0, len(bound_coords)):
            s.addelectricalcontact()
            s.set("name", f"b{i}")
            s.set("surface type", "coordinates of domain")
            s.eval(f'set("coordinates", {{{[float(coord) for coord in bound_coords[i]]}}});')

            # Add boundary condition settings
            self.boundary_condition_settings[f"b{i}"] = {}
        if boundary_settings:
            self.set_boundary_conditions(boundary_settings=boundary_settings)

        s.addsurfacerecombinationbc()
        s.set("electron velocity", ss.electron_velocity * cm)
        s.set("hole velocity", ss.hole_velocity * cm)
        s.set("surface type", "material:material")
        s.set("material 1", self.layerstack.get_layer_to_material()[ss.metal_layer])
        s.set("material 2", self.layerstack.get_layer_to_material()[ss.dopant_layer])

    def set_boundary_conditions(self, boundary_settings: dict[str, dict]):
        """
        Set electrical boundary condition settings

        Parameters:
            boundary_settings: Mapping between boundary condition name to its electrical boundary condition settings.
                                Ex. {
                                        "b0":
                                        {
                                            "bc mode": "steady state",
                                            "sweep type": "single",
                                            "voltage": 1.1,
                                            "force ohmic": True
                                        },
                                        "b1":
                                        {
                                            "bc mode": "steady state",
                                            "sweep type": "single",
                                            "voltage": 0.1,
                                            "force ohmic": True
                                        }
                                    }

        """
        s = self.session
        for name, settings in boundary_settings.items():
            try:
                s.select(f"::model::CHARGE::boundary conditions::{name}")

                if settings.get("name", None):
                    if settings["name"] in self.boundary_condition_settings:
                        raise KeyError(
                            f"'{settings['name']}' found in existing boundary conditions. Cannot swap "
                            + f"'{name}' boundary name with '{settings['name']}' name."
                        )

                    if (
                        "+" in settings["name"]
                        or "-" in settings["name"]
                        or "_" in settings["name"]
                    ):
                        orig_name = settings["name"]
                        settings["name"] = (
                            settings["name"]
                            .replace("+", "")
                            .replace("-", "")
                            .replace("-", "")
                        )
                        logger.warning(
                            f"Boundary condition name changed from '{orig_name}' to '{settings['name']}'. Names cannot have +, -, or _ symbols."
                        )

                    # Override the boundary condition name while retaining its settings
                    self.boundary_condition_settings[
                        settings["name"]
                    ] = self.boundary_condition_settings.pop(name)
            except Exception as err:
                logger.warning(
                    f"{err}\nCannot find {name} boundary, skipping settings for this boundary."
                )
                continue
            for setting, value in settings.items():
                try:
                    s.set(setting, value)
                    if settings.get("name", None):
                        self.boundary_condition_settings[settings["name"]][setting] = value
                    else:
                        self.boundary_condition_settings[name][setting] = value
                except Exception as err:
                    logger.warning(
                        f"{err}\nCannot find {setting} setting, skipping setting."
                    )
                    continue

    def add_charge_monitor(self):
        s = self.session
        ss = self.simulation_settings

        s.addchargemonitor()
        s.set("x", ss.x * um)
        s.set("y", ss.y * um)
        s.set("z", ss.z * um)
        s.set("x span", ss.xspan * um)
        s.set("y span", ss.yspan * um)
        s.set("z span", ss.zspan * um)
        s.set("monitor type", ss.dimension)
        s.set("integrate total charge", True)
        s.set("save data", True)
        s.set("filename", str(self.simulation_dirpath.resolve() / "charge.mat"))

