import os
import hashlib
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path, PosixPath, WindowsPath
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement

import pydantic
from gdsfactory.component import Component
from gdsfactory import logger
from gdsfactory.pdk import get_layer_stack
from gdsfactory.technology import LayerStack, LogicalLayer, DerivedLayer
from gdsfactory.typings import PathType
from gdsfactory import get_layer
from gdsfactory.technology.processes import Etch, Grow, Anneal, Planarize, ImplantGaussian, ImplantPhysical, DopingConstant

from gplugins.lumerical.config import um


def to_lbr(
    material_map: dict[str, str],
    layerstack: LayerStack | None = None,
    process: tuple | None = None,
    dirpath: PathType | None = "",
    use_pdk_material_names: bool = False,
) -> Path:
    """
    Generate an XML file representing a Lumerical Layer Builder process file based on provided material map.

    Args:
        material_map: A dictionary mapping materials used in the layer stack to Lumerical materials.
        layerstack: Layer stack that has info on layer names, layer numbers, thicknesses, etc.
        process: Process (etch, grow, implant, etc.) that affects layerstack
        dirpath: Directory to save process file (process.lbr)
        use_pdk_material_names: Use PDK material names in the pattern material and background material fields.
                                This is mainly used for DEVICE simulations where several materials are grouped together
                                in one material to describe electrical, thermal, and optical properties.

    Returns:
        Process file path

    Notes:
        This function generates an XML file representing a Layer Builder file for Lumerical, based on the provided the active PDK
        and material map. It creates 'process.lbr' in the current working directory, containing layer information like name,
        material, thickness, sidewall angle, and other properties specified in the layer stack.
    """
    layerstack = layerstack or get_layer_stack()
    layer_to_layername = layerstack.get_layer_to_layername()

    # Extract processed layers, these are considered grow or dopant layers
    layer_to_processes = {p.layer: [] for p in process if hasattr(p, "layer") and p.layer}
    for p in process:
        if hasattr(p, "layer") and p.layer:
            layer_to_processes[p.layer].append(p)

    layer_builder = Element("layer_builder")

    process_name = SubElement(layer_builder, "process_name")
    process_name.text = ""

    layers = SubElement(layer_builder, "layers")
    for layer_name, layer_level in layerstack.layers.items():
        layer = SubElement(layers, "layer")

        try:
            layer_number = layer_level.derived_layer.layer.layer if layer_level.derived_layer \
                else layer_level.layer.layer.layer
            layer_datatype = layer_level.derived_layer.layer.datatype if layer_level.derived_layer \
                else layer_level.layer.layer.datatype
        except:
            raise(TypeError, f"Layer {layer_name} must be of type LogicalLayer.")



        # Default params
        layer_params = {
            "enabled": "1",
            "pattern_alpha": "0.8",
            "start_position_auto": "0",
            "background_alpha": "0.3",
            "pattern_material_index": "0",
            "material_index": "0",
            "name": layer_name,
            "layer_name": f'{layer_number}:{layer_datatype}',
            "start_position": f'{layer_level.zmin * um}',
            "thickness": f'{layer_level.thickness * um}',
            "sidewall_angle": f'{90 - layer_level.sidewall_angle}',
            "pattern_growth_delta": f"{layer_level.bias * um}"
            if layer_level.bias
            else "0",
        }

        for param, val in layer_params.items():
            layer.set(param, val)


        layer_processes = layer_to_processes.get((layer_number, layer_datatype), None)
        if layer_processes and any(type(p) is Etch or type(p) is Grow for p in layer_processes):
            layer.set(
                "pattern_material",
                f'{material_map.get(layer_level.material, "")}'
                if not use_pdk_material_names
                else layer_level.material,
            )
            layer.set("process", "Grow")
        else:
            layer.set(
                "material",
                f'{material_map.get(layer_level.material, "")}'
                if not use_pdk_material_names
                else layer_level.material,
            )
            layer.set("process", "Background")


    doping_layers = SubElement(layer_builder, "doping_layers")
    for gds_layer, layer_processes in layer_to_processes.items():
        if any(type(p) is ImplantPhysical or type(p) is ImplantGaussian or type(p) is DopingConstant for p in layer_processes):
            for p in layer_processes:
                affected_layers = p.layers_or if p.layers_or else []
                affected_layers += p.layers_diff if p.layers_diff else []
                affected_layers += p.layers_and if p.layers_and else []
                affected_layers += p.layers_xor if p.layers_xor else []

                for affected_layer in affected_layers:
                    layer_level = layerstack.layers.get(layer_to_layername[affected_layer], None)
                    try:
                        layer_number = layer_level.derived_layer.layer.layer if layer_level.derived_layer \
                            else layer_level.layer.layer.layer
                        layer_datatype = layer_level.derived_layer.layer.datatype if layer_level.derived_layer \
                            else layer_level.layer.layer.datatype
                    except:
                        raise (TypeError,
                               f"Layer {layer_name} must be of type LogicalLayer.")

                    # Handle dopant range (depth of penetration)
                    if type(p) is ImplantGaussian:
                        dopant_range = p.range * um
                    elif type(p) is DopingConstant:
                        dopant_range = (p.zmax - p.zmin) / 2 * um
                    else:
                        dopant_range = layer_level.thickness / 2 * um

                    # Handle concentration
                    if type(p) is ImplantGaussian or type(p) is DopingConstant:
                        concentration = p.peak_conc
                    elif type(p) is ImplantPhysical:
                        concentration = p.dose
                    else:
                        concentration = 0

                    # Handle ion type p or n
                    if p.ion == "n" or p.ion == "p":
                        ion = p.ion
                    else:
                        raise ValueError( f'Dopant must be "p" or "n". Got {p.ion}.')

                    doping_layer = SubElement(doping_layers, "layer")
                    doping_params = {
                        "z_surface_positions": f'{(layer_level.zmin + layer_level.thickness) * um}',
                        "distribution_function": "Gaussian",
                        "phi": f"{p.twist}" if type(p) is ImplantPhysical and p.twist else "0",
                        "lateral_scatter": f"{p.lateral_straggle * um}" if type(p) is ImplantGaussian and p.lateral_straggle else "2e-8",
                        "range": f"{dopant_range}",
                        "theta": f"{p.tilt}" if type(p) is ImplantPhysical and p.tilt else "0",
                        "mask_layer_number": f'{layer_number}:{layer_datatype}',
                        "kurtosis": "0",
                        "process": "Implant",
                        "skewness": "0",
                        "straggle": f"{p.vertical_straggle * um}" if type(p) is ImplantGaussian else f"{layer_level.thickness}",
                        "concentration": f"{concentration}",
                        "enabled": "1",
                        "name": f"{p.name}",
                        "dopant": f"{ion}",
                    }
                    for param, val in doping_params.items():
                        doping_layer.set(param, val)

    # If no doping layers exist, delete element
    if len(doping_layers) == 0:
        layer_builder.remove(doping_layers)

    # Prettify XML
    rough_string = ET.tostring(layer_builder, "utf-8", short_empty_elements=False)
    reparsed = minidom.parseString(rough_string)
    xml_str = reparsed.toprettyxml(indent="  ")

    if dirpath:
        process_file_path = Path(str(dirpath.resolve())) / "process.lbr"
    else:
        process_file_path = Path(__file__).resolve().parent / "process.lbr"
    with open(str(process_file_path), "w") as f:
        f.write(xml_str)

    return process_file_path


def draw_geometry(
    session: object,
    gdspath: PathType,
    process_file_path: PathType,
) -> None:
    """
    Draw geometry in Lumerical simulator

    Parameters:
        session: Lumerical session
        gdspath: GDS path
        process_file_path: Process file path
    """
    s = session
    s.addlayerbuilder()
    s.set("x", 0)
    s.set("y", 0)
    s.set("z", 0)
    s.loadgdsfile(str(gdspath))
    try:
        s.loadprocessfile(str(process_file_path))
    except Exception as err:
        raise Exception(
            f"{err}\nProcess file cannot be imported. Likely causes are dopants in the process file or syntax errors."
        ) from err


class Results:
    """
    Results are stored in this dynamic class. Any type of results can be stored.

    This class allows designers to arbitrarily add results. Results are pickled to be saved onto working system.
    Results can be retrieved via unpickling.
    """

    def __init__(self, prefix: str = "", dirpath: Path | None = None, **kwargs):
        if isinstance(dirpath, str):
            dirpath = Path(dirpath)
        self.dirpath = dirpath or Path(".")
        self.prefix = prefix
        for key, value in kwargs.items():
            setattr(self, key, value)

    def save_pickle(self, dirpath: Path | None = None):
        """
        Save results by pickling as *_results.pkl file

        Parameters:
            dirpath: Directory to store pickle file
        """
        if dirpath is None:
            with open(str(self.dirpath.resolve() / f"{self.prefix}_results.pkl"), "wb") as f:
                pickle.dump(self, f)
                logger.info(f"Cached results to {self.dirpath} -> {self.prefix}_results.pkl")
        else:
            with open(str(dirpath.resolve() / f"{self.prefix}_results.pkl"), "wb") as f:
                pickle.dump(self, f)
                logger.info(f"Cached results to {dirpath} -> {self.prefix}_results.pkl")

    def get_pickle(self, dirpath: Path | None = None) -> object:
        """
        Get results from *_results.pkl file

        Parameters:
            dirpath: Directory to get pickle file

        Returns:
            Results as an object with results
        """
        if isinstance(dirpath, str):
            dirpath = Path(dirpath)
        if dirpath is None:
            with open(str(self.dirpath.resolve() / f"{self.prefix}_results.pkl"), "rb") as f:
                unpickler = PathUnpickler(f)
                results = unpickler.load()
                if not results.dirpath == self.dirpath:
                    results.dirpath = self.dirpath
                logger.info(f"Recalled results from {self.dirpath} -> {self.prefix}_results.pkl")
        else:
            with open(str(dirpath.resolve() / f"{self.prefix}_results.pkl"), "rb") as f:
                unpickler = PathUnpickler(f)
                results = unpickler.load()
                if not results.dirpath == dirpath:
                    results.dirpath = dirpath
                logger.info(f"Recalled results from {dirpath} -> {self.prefix}_results.pkl")

        return results

    def available(self, dirpath: Path | None = None) -> bool:
        """
        Check if '*_results.pkl' file exists and results can be loaded

        Parameters:
            dirpath: Directory with pickle file

        Returns:
            True if results exist, False otherwise.
        """
        if isinstance(dirpath, str):
            dirpath = Path(dirpath)
        if dirpath is None:
            results_file = self.dirpath.resolve() / f"{self.prefix}_results.pkl"
        else:
            results_file = dirpath.resolve() / f"{self.prefix}_results.pkl"
        return results_file.is_file()


class Simulation:
    """
    Represents the simulation object used to simulate GDSFactory devices.

    This simulation object's primary purpose is to reduce time simulating by recalling hashed results.
    """

    # the hash of the system last time convergence was executed
    last_hash: int = -1

    # A dynamic object used to store convergence results
    convergence_results: Results

    def __init__(
        self,
        component: Component,
        layerstack: LayerStack | None = None,
        simulation_settings: pydantic.BaseModel | None = None,
        convergence_settings: pydantic.BaseModel | None = None,
        dirpath: Path | None = None,
    ):
        self.dirpath = dirpath or Path(".")
        self.component = component
        self.layerstack = layerstack or get_layer_stack()
        self.simulation_settings = simulation_settings
        self.convergence_settings = convergence_settings

        self.last_hash = hash(self)

        # Create directory for simulation files
        self.simulation_dirpath = (
            self.dirpath / f"{self.__class__.__name__}_{self.last_hash}"
        )
        self.simulation_dirpath.mkdir(parents=True, exist_ok=True)

        # Create attribute for convergence results
        self.convergence_results = Results(
            prefix="convergence", dirpath=self.simulation_dirpath
        )

    def __hash__(self) -> int:
        """
        Returns a hash of all state this Simulation contains
        Subclasses should include functionality-specific state (e.g. convergence info) here.
        This is used to determine simulation convergence (i.e. if it needs to be rerun)

        Hashed items:
        - component
        - layer stack
        - simulation settings
        - convergence settings
        """
        h = hashlib.sha1()
        if self.component is not None:
            try:
                h.update(self.component.hash())
            except:
                # Return port layers back to integers
                for i in range(0, len(self.component.ports)):
                    self.component.ports[i].layer = get_layer(self.component.ports[i].layer).layer

                for i in range(0, len(self.component.references)):
                    for j in range(0, len(self.component.references[i].cell.ports)):
                        self.component.references[i].cell.ports[j].layer = get_layer(
                            self.component.references[i].cell.ports[j].layer)

                h.update(self.component.hash())

        if self.layerstack is not None:
            h.update(self.layerstack.model_dump_json().encode("utf-8"))
        if self.simulation_settings is not None:
            h.update(self.simulation_settings.model_dump_json().encode("utf-8"))
        if self.convergence_settings is not None:
            h.update(self.convergence_settings.model_dump_json().encode("utf-8"))
        return int.from_bytes(h.digest(), "big")

    def convergence_is_fresh(self) -> bool:
        """
        Returns if this simulation needs to be re-run.
        This could be caused by this simulation's
        configuration being changed.
        """
        return hash(self) == self.last_hash

    def load_convergence_results(self):
        """
        Loads convergence results from pickle file into class attribute
        """
        self.convergence_results = self.convergence_results.get_pickle()

    def save_convergence_results(self):
        """
        Saves convergence_results to pickle file while adding setup information and resultant accurate simulation settings.
        This includes:
        - component hash
        - layerstack
        - convergence_settings
        - simulation_settings

        This is usually done after convergence testing is completed and simulation settings are accurate and should be
        saved for future reference/recall.
        """
        self.convergence_results.convergence_settings = self.convergence_settings
        self.convergence_results.simulation_settings = self.simulation_settings
        self.convergence_results.component_hash = self.component.hash()
        self.convergence_results.layerstack = self.layerstack
        self.simulation_dirpath.mkdir(parents=True, exist_ok=True)
        self.convergence_results.save_pickle()

    def is_same_convergence_results(self) -> bool:
        """
        Returns whether convergence results' setup are the same as the current setup for the simulation.
        This is important for preventing hash collisions.
        """
        try:
            return (
                self.convergence_results.convergence_settings
                == self.convergence_settings
                and self.convergence_results.component_hash
                == self.component.hash_geometry()
                and self.convergence_results.layerstack == self.layerstack
            )
        except AttributeError:
            return False


class PathUnpickler(pickle.Unpickler):
    """
    Unpickles objects while handling OS-dependent paths
    """
    def find_class(self, module, name):
        if module == 'pathlib' and (name == 'PosixPath' or name == "WindowsPath"):
            return WindowsPath if os.name == 'nt' else PosixPath
        return super().find_class(module, name)