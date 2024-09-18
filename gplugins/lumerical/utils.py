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
from gdsfactory.technology import LayerStack
from gdsfactory.typings import PathType
from gdsfactory import get_layer

from gplugins.lumerical.config import ENABLE_DOPING, um


def layerstack_to_lbr(
    material_map: dict[str, str],
    layerstack: LayerStack | None = None,
    dirpath: PathType | None = "",
    use_pdk_material_names: bool = False,
) -> Path:
    """
    Generate an XML file representing a Lumerical Layer Builder process file based on provided material map.

    Args:
        material_map: A dictionary mapping materials used in the layer stack to Lumerical materials.
        layerstack: Layer stack that has info on layer names, layer numbers, thicknesses, etc.
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

    layer_builder = Element("layer_builder")

    process_name = SubElement(layer_builder, "process_name")
    process_name.text = ""

    layers = SubElement(layer_builder, "layers")
    doping_layers = SubElement(layer_builder, "doping_layers")
    for layer_name, layer_info in layerstack.to_dict().items():
        if layer_info["info"].get("layer_type", "") == "grow":
            process = "Grow"
        elif layer_info["info"].get("layer_type", "")  == "background":
            process = "Background"
        elif (
            layer_info["info"].get("layer_type", "")  == "doping"
            or layer_info["info"].get("layer_type", "")  == "implant"
        ):
            process = "Implant"
        else:
            logger.warning(
                f'"{layer_info["info"].get("layer_type", "") }" layer type not supported for "{layer_name}" in Lumerical. Defaulting to "grow".'
            )
            process = "Grow"

        ### Set optical and metal layers
        if process == "Grow" or process == "Background":
            layer = SubElement(layers, "layer")

            try:
                gds_layer = f'{layer_info["layer"].layer[0]}:{layer_info["layer"].layer[1]}'
            except ValueError:
                gds_layer = f'{layer_info["layer"].layer.layer}:{layer_info["layer"].layer.datatype}'
            except AttributeError:
                gds_layer = f'{layerstack.layers[layer_name].derived_layer.layer.layer}:{layerstack.layers[layer_name].derived_layer.layer.datatype}'

            # Default params
            layer_params = {
                "enabled": "1",
                "pattern_alpha": "0.8",
                "start_position_auto": "0",
                "background_alpha": "0.3",
                "pattern_material_index": "0",
                "material_index": "0",
                "name": layer_name,
                "layer_name": gds_layer,
                "start_position": f'{layer_info["zmin"] * um}',
                "thickness": f'{layer_info["thickness"] * um}',
                "process": f"{process}",
                "sidewall_angle": f'{90 - layer_info["sidewall_angle"]}',
                "pattern_growth_delta": f"{layer_info['bias'] * um}"
                if layer_info["bias"]
                else "0",
            }

            for param, val in layer_params.items():
                layer.set(param, val)

            if process == "Grow":
                layer.set(
                    "pattern_material",
                    f'{material_map.get(layer_info["material"], "")}'
                    if not use_pdk_material_names
                    else layer_info["material"],
                )
            elif process == "Background":
                layer.set(
                    "material",
                    f'{material_map.get(layer_info["material"], "")}'
                    if not use_pdk_material_names
                    else layer_info["material"],
                )

        if (process == "Implant" or process == "Background") and ENABLE_DOPING:
            ### Set doping layers
            # KNOWN ISSUE: If a metal or optical layer has the same name as a doping layer, Layer Builder will not compile
            # the process file correctly and the doping layer will not appear. Therefore, doping layer names MUST be unique.
            # FIX: Appending "_doping" to name

            # KNOWN ISSUE: If the 'process' is not 'Background' or 'Implant' for dopants, this will crash CHARGE upon importing process file.
            # FIX: Ensure process is Background or Implant before proceeding to create entry

            # KNOWN ISSUE: Dopant must be either 'p' or 'n'. Anything else will cause CHARGE to crash upon importing process file.
            # FIX: Raise ValueErrorr when dopant is specified incorrectly
            if layer_info["info"].get(
                "background_doping_concentration", False
            ) and layer_info["info"].get("background_doping_ion", False):
                doping_layer = SubElement(doping_layers, "layer")
                doping_params = {
                    "z_surface_positions": f'{layer_info["zmin"] * um}',
                    "distribution_function": "Gaussian",
                    "phi": "0",
                    "lateral_scatter": "2e-08",
                    "range": f"{layer_info['thickness'] / 2 * um}",
                    "theta": "0",
                    "mask_layer_number": f'{layer_info["layer"].layer[0]}:{layer_info["layer"].layer[1]}',
                    "kurtosis": "0",
                    "process": f"{process}",
                    "skewness": "0",
                    "straggle": f"{layer_info['thickness'] * um}",
                    "concentration": f"{layer_info['info']['background_doping_concentration']}",
                    "enabled": "1",
                    "name": f"{layer_name}_doping",
                }
                for param, val in doping_params.items():
                    doping_layer.set(param, val)

                if (
                    layer_info["info"]["background_doping_ion"] == "n"
                    or layer_info["info"]["background_doping_ion"] == "p"
                ):
                    doping_layer.set("dopant", layer_info["info"]["background_doping_ion"])
                else:
                    raise ValueError(
                        f'Dopant must be "p" or "n". Got {layer_info["info"]["background_doping_ion"]}.'
                    )

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