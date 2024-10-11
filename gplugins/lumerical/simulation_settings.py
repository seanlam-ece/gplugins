from typing import Literal

from gdsfactory.cross_section import LayerSpec
from pydantic import BaseModel

simulation_settings_fdtd = [
    "allow grading in x",
    "allow grading in y",
    "allow grading in z",
    "allow symmetry on all boundaries",
    "always use complex fields",
    "angle phi",
    "angle theta",
    "auto shutoff max",
    "auto shutoff min",
    "background material",
    "bfast alpha",
    "bfast dt multiplier",
    "bloch units",
    "checkpoint at shutoff",
    "checkpoint during simulation",
    "checkpoint period",
    "conformal meshing refinement",
    "define x mesh by",
    "define y mesh by",
    "define z mesh by",
    "dimension",
    "direction",
    "down sample time",
    "dt",
    "dt stability factor",
    "dx",
    "dy",
    "dz",
    "enabled",
    "extend structure through pml",
    "force symmetric x mesh",
    "force symmetric y mesh",
    "force symmetric z mesh",
    "global monitor custom frequency samples",
    "global monitor frequency center",
    "global monitor frequency points",
    "global monitor frequency span",
    "global monitor maximum frequency",
    "global monitor maximum wavelength",
    "global monitor minimum frequency",
    "global monitor minimum wavelength",
    "global monitor sample spacing",
    "global monitor use source limits",
    "global monitor use wavelength spacing",
    "global monitor wavelength center",
    "global monitor wavelength span",
    "global source bandwidth",
    "global source center frequency",
    "global source center wavelength",
    "global source eliminate discontinuities",
    "global source frequency",
    "global source frequency span",
    "global source frequency start",
    "global source frequency stop",
    "global source offset",
    "global source optimize for short pulse",
    "global source pulse type",
    "global source pulselength",
    "global source set frequency",
    "global source set time domain",
    "global source set wavelength",
    "global source wavelength span",
    "global source wavelength start",
    "global source wavelength stop",
    "grading factor",
    "index",
    "injection axis",
    "kx",
    "ky",
    "kz",
    "max source time signal length",
    "mesh accuracy",
    "mesh allowed size increase factor",
    "mesh cells per wavelength",
    "mesh cells x",
    "mesh cells y",
    "mesh cells z",
    "mesh distance between fixed points",
    "mesh frequency max",
    "mesh frequency min",
    "mesh merge distance",
    "mesh minimum neighbor size",
    "mesh refinement",
    "mesh size reduction factor",
    "mesh step for metals",
    "mesh type",
    "mesh wavelength max",
    "mesh wavelength min",
    "meshing refinement",
    "meshing tolerance",
    "min mesh step",
    "nx",
    "ny",
    "nz",
    "override simulation bandwidth for mesh generation",
    "param1",
    "param2",
    "pml alpha",
    "pml alpha polynomial",
    "pml kappa",
    "pml layers",
    "pml max layers",
    "pml min layers",
    "pml polynomial",
    "pml profile",
    "pml sigma",
    "pml type",
    "same settings on all boundaries",
    "set based on source angle",
    "set process grid",
    "set simulation bandwidth",
    "simulation frequency max",
    "simulation frequency min",
    "simulation temperature",
    "simulation time",
    "simulation wavelength max",
    "simulation wavelength min",
    "snap pec to yee cell boundary",
    "source index",
    "type",
    "use auto shutoff",
    "use bfast fdtd",
    "use divergence checking",
    "use early shutoff",
    "use legacy conformal interface detection",
    "use mesh step for metals",
    "use relative coordinates",
]


material_name_to_lumerical_default = {
    "si": "Si (Silicon) - Palik",
    "sio2": "SiO2 (Glass) - Palik",
    "sin": "Si3N4 (Silicon Nitride) - Phillip",
    "tungsten": "W (Tungsten) - Palik",
    "cu": "Cu (Copper) - CRC",
    "air": "Air",
    "TiN": "TiN - Palik",
    "Aluminum": "Al (Aluminium) - Palik",
}

material_name_to_lumerical_ele_therm_default = {
    "si": "Si (Silicon)",
    "sio2": "SiO2 (Glass) - Sze",
    "sin": "Si3N4 (Silicon nitride) - Sze",
    "tungsten": "W (Tungsten) - Palik",
    "cu": "Cu (Copper) - CRC",
    "air": "Air",
    "TiN": "TiN - Palik",
    "Aluminum": "Al (Aluminium) - CRC",
}


class SimulationSettingsLumericalEme(BaseModel):
    """Lumerical EME simulation_settings.

    Parameters:
        wavelength: Wavelength (um)
        wavelength_start: Starting wavelength in wavelength range (um)
        wavelength_stop: Stopping wavelength in wavelength range (um)
        material_fit_tolerance: Material fit coefficient
        material_name_to_lumerical: Mapping of PDK materials to Lumerical materials
        group_cells: Number of cells in each group
        group_spans: Span size in each group (um)
        group_subcell_methods: Methods to analyze each cross section
        num_modes: Number of modes
        energy_conservation: Ensure results are passive or conserve energy.
        mesh_cells_per_wavelength: Number of mesh cells per wavelength
        ymin_boundary: y min boundary condition
        ymax_boundary: y max boundary condition
        zmin_boundary: z min boundary condition
        zmax_boundary: z max boundary condition
        port_extension: Port extension beyond the simulation boundary (um)
        pml_layers: Number of PML layers used if PML boundary conditions used.
        ymargin: Y margin from component to simulation boundary (um)
        zmargin: Z margin from component to simulation boundary (um)
    """

    wavelength: float = 1.55
    wavelength_start: float = 1.5
    wavelength_stop: float = 1.6
    material_fit_tolerance: float = 0.001
    material_name_to_lumerical: dict[str, str] = material_name_to_lumerical_default

    group_cells: list[int] = [1, 30, 1]
    group_subcell_methods: list[Literal["CVCS"] | None] = [None, "CVCS", None]
    num_modes: int = 30
    energy_conservation: Literal[
        "make passive", "conserve energy"
    ] | None = "make passive"

    mesh_cells_per_wavelength: int = 60

    ymin_boundary: Literal[
        "Metal", "PML", "Anti-Symmetric", "Symmetric"
    ] = "Anti-Symmetric"
    ymax_boundary: Literal["Metal", "PML", "Anti-Symmetric", "Symmetric"] = "Metal"
    zmin_boundary: Literal["Metal", "PML", "Anti-Symmetric", "Symmetric"] = "Metal"
    zmax_boundary: Literal["Metal", "PML", "Anti-Symmetric", "Symmetric"] = "Metal"

    port_extension: float = 1.0

    pml_layers: int = 12

    ymargin: float = 2.0
    zmargin: float = 1.0

    class Config:
        """pydantic basemodel config."""

        arbitrary_types_allowed = True


LUMERICAL_EME_SIMULATION_SETTINGS = SimulationSettingsLumericalEme()


class SimulationSettingsLumericalFdtd(BaseModel):
    """Lumerical FDTD simulation_settings.

    Parameters:
        solver_type: CPU or GPU type solver
        port_margin: on both sides of the port width (um).
        port_extension: port extension that extends waveguide passed ports (um).
        port_translation: Translate port by x microns away from (positive) or toward (negative) the device.
        mesh_accuracy: 2 (1: coarse, 2: fine, 3: superfine).
        material_fit_tolerance: Material fit coefficient
        zmargin: for the FDTD region (um).
        ymargin: for the FDTD region (um).
        xmargin: for the FDTD region (um).
        wavelength_start: 1.2 (um).
        wavelength_stop: 1.6 (um).
        wavelength_points: wavelength points for sparams.
        simulation_time: (s) related to max path length
            3e8/2.4*10e-12*1e6 = 1.25mm.
        simulation_temperature: in kelvin (default = 300).
        frequency_dependent_profile: compute mode profiles for each wavelength.
        field_profile_samples: number of wavelengths to compute field profile.
        material_name_to_lumerical: Mapping of PDK materials to Lumerical materials
        port_field_intensity_threshold: E-field intensity at the edge of each port. Used to resize ports and FDTD region.
    """
    solver_type: Literal["cpu", "gpu"] = "cpu"
    port_margin: float = 0.2
    port_extension: float = 5.0
    port_translation: float = 0.0
    material_fit_tolerance: float = 0.001
    mesh_accuracy: int = 4
    wavelength_start: float = 1.5
    wavelength_stop: float = 1.6
    wavelength_points: int = 200
    simulation_time: float = 10e-12
    simulation_temperature: float = 300
    frequency_dependent_profile: bool = True
    field_profile_samples: int = 15
    distance_monitors_to_pml: float = 0.5
    material_name_to_lumerical: dict[str, str] = material_name_to_lumerical_default
    port_field_intensity_threshold: float = 1e-6

    class Config:
        """pydantic basemodel config."""

        arbitrary_types_allowed = True


SIMULATION_SETTINGS_LUMERICAL_FDTD = SimulationSettingsLumericalFdtd()


class SimulationSettingsLumericalCharge(BaseModel):
    """Lumerical CHARGE simulation_settings.

    Parameters:
        solver_mode: CHARGE solver mode
        simulation_temperature: Temperature in K
        temperature_dependence: Set this to have temperature dependent simulation (cross coupling)
        min_edge_length: Minimum edge length in um for a triangle or tetrahedron mesh
        max_edge_length: Maximum edge length in um  for a triangle or tetrahedron mesh
        max_refine_steps: Maximum number of vertices that can be added to the mesh at each mesh refinement stage
        min_time_step: Minimum time step in seconds (only for transient simulations)
        max_time_step: Maximum time step in seconds (only for transient simulations)
        norm_length: Length of device in perpendicular direction of simulation region, usually used to calculate current
                    from the current density. Units of microns.
        vac_amplitude: Small signal AC voltage amplitude in V (only for small signal AC simulations)
        frequency_spacing: Type of frequency sampling (only for small signal AC simulations)
        frequency: AC frequency to simulate
        start_frequency: Start frequency in Hz
        stop_frequency: Stop frequency in Hz
        num_frequency_pts: Number of frequency points to consider. If frequency_spacing = linear, num_frequency_pts
                            represents the number of points from start_frequency to stop_frequency.
                            If frequency_spacing = log, num_frequency_pts represents the number of frequency points per
                            decade.
        dimension: Dimension and direction of simulation region
        xmin_boundary: "closed" = simulation bounds defined by simulation region geometry. "open" = simulation bounds
                        defined by the structure shape. "shell" = similar to closed but adds additional shell layers.
        xmax_boundary: "closed" = simulation bounds defined by simulation region geometry. "open" = simulation bounds
                        defined by the structure shape. "shell" = similar to closed but adds additional shell layers.
        ymin_boundary: "closed" = simulation bounds defined by simulation region geometry. "open" = simulation bounds
                        defined by the structure shape. "shell" = similar to closed but adds additional shell layers.
        ymax_boundary: "closed" = simulation bounds defined by simulation region geometry. "open" = simulation bounds
                        defined by the structure shape. "shell" = similar to closed but adds additional shell layers.
        zmin_boundary: "closed" = simulation bounds defined by simulation region geometry. "open" = simulation bounds
                        defined by the structure shape. "shell" = similar to closed but adds additional shell layers.
        zmax_boundary: "closed" = simulation bounds defined by simulation region geometry. "open" = simulation bounds
                        defined by the structure shape. "shell" = similar to closed but adds additional shell layers.
        x: Center x coordinate of simulation region (um)
        y: Center y coordinate of simulation region (um)
        z: Center z coordinate of simulation region (um)
        xspan: X span for simulation region (um)
        yspan: Y span for simulation region (um)
        zspan: Z span for simulation region (um)
        material_fit_tolerance: Material fit coefficient
        optical_material_name_to_lumerical: Optical material mapping between PDK materials and Lumerical materials
        ele_therm_material_name_to_lumerical: Electrical and thermal material mapping between PDK materials and Lumerical materials
        metal_layer: Layer used for electrical boundary conditions.
        dopant_layer: Layer used for surface recombination boundary condition. This layer interfaces with the metal_layer.
        electron_velocity: Electron velocity for surface recombination boundary (cm/s)
        hole_velocity: Hole velocity for surface recombination boundary (cm/s)

    """

    solver_mode: Literal["steady state", "transient", "ssac"] = "steady state"
    simulation_temperature: float = 300
    temperature_dependence: Literal[
        "isothermal", "non-isothermal", "coupled"
    ] = "isothermal"
    min_edge_length: float = 1e-3
    max_edge_length: float = 1
    max_refine_steps: int = 20e3
    min_time_step: float = 1e-12
    max_time_step: float = 1e-6
    norm_length: float = 1.0
    vac_amplitude: float = 1e-3
    frequency_spacing: Literal["single", "linear", "log"] = "single"
    frequency: float = 1e6
    start_frequency: float = 1e6
    stop_frequency: float = 2e6
    num_frequency_pts: int = 100
    dimension: Literal["2D X-Normal", "2D Y-Normal", "3D"] = "2D Y-Normal"
    xmin_boundary: Literal["closed", "open", "shell"] = "closed"
    xmax_boundary: Literal["closed", "open", "shell"] = "closed"
    ymin_boundary: Literal["closed", "open", "shell"] = "closed"
    ymax_boundary: Literal["closed", "open", "shell"] = "closed"
    zmin_boundary: Literal["closed", "open", "shell"] = "closed"
    zmax_boundary: Literal["closed", "open", "shell"] = "closed"

    x: float = 0
    y: float = 0
    z: float = 0
    xspan: float = 5.0
    yspan: float = 5.0
    zspan: float = 1.0

    material_fit_tolerance: float = 1e-3
    optical_material_name_to_lumerical: dict[
        str, str
    ] = material_name_to_lumerical_default
    ele_therm_material_name_to_lumerical: dict[
        str, str
    ] = material_name_to_lumerical_ele_therm_default

    metal_layer: LayerSpec = (40, 0)
    dopant_layer: LayerSpec = (3, 0)

    # Boundary conditions
    electron_velocity: float = 1e7
    hole_velocity: float = 1e7

    class Config:
        """pydantic basemodel config."""

        arbitrary_types_allowed = True


LUMERICAL_CHARGE_SIMULATION_SETTINGS = SimulationSettingsLumericalCharge()

class SimulationSettingsLumericalMode(BaseModel):
    """
    Lumerical MODE simulation settings.

    Parameters:
        injection_axis: Normal direction of simulation plane
        mesh_cells_per_wavl: Number of mesh cells per wavelength
        wavl: Center wavelength (um)
        wavl_start: Start wavelength (um)
        wavl_end: End wavelength (um)
        wavl_pts: Number of wavelength points
        num_modes: Number of modes
        target_mode: Target mode number. Fundamental mode starts at 1.
        x: Center x coordinate of simulation region (um)
        y: Center y coordinate of simulation region (um)
        z: Center z coordinate of simulation region (um)
        xspan: X span for simulation region (um)
        yspan: Y span for simulation region (um)
        zspan: Z span for simulation region (um)
        xmin_boundary: xmin boundary condition
        xmax_boundary: xmax boundary condition
        ymin_boundary: ymin boundary condition
        ymax_boundary: ymax boundary condition
        zmin_boundary: zmin boundary condition
        zmax_boundary: zmax boundary condition
        pml_layers: Number of PML layers
        include_dispersion: If True, include dispersion calculations
        material_fit_tolerance: Material fit tolerance
        material_name_to_lumerical: Mapping of PDK materials to Lumerical materials
        is_bent_waveguide: If True, include bent waveguide simulation
        bend_radius: Bend radius (um)
        bend_location: Bend location
        bend_location_x: x location of bend
        bend_location_y: y location of bend
        bend_location_z: z location of bend
        efield_intensity_threshold: E-field intensity threshold at the edge of the simulation region
    """
    injection_axis: Literal["2D X normal", "2D Y normal"] = "2D X normal"

    mesh_cells_per_wavl: int = 50

    wavl: float = 1.55
    wavl_start: float = 1.5
    wavl_end: float = 1.6
    wavl_pts: int = 20
    num_modes: int = 20
    target_mode: int = 1

    # Simulation region to device edge margins
    x: float = 0
    y: float = 0
    z: float = 0
    xspan: float = 5.0
    yspan: float = 5.0
    zspan: float = 1.0

    # Boundary conditions
    xmin_boundary: Literal["Metal", "PML"] = "Metal"
    xmax_boundary: Literal["Metal", "PML"] = "Metal"
    ymin_boundary: Literal["Metal", "PML"] = "Metal"
    ymax_boundary: Literal["Metal", "PML"] = "Metal"
    zmin_boundary: Literal["Metal", "PML"] = "Metal"
    zmax_boundary: Literal["Metal", "PML"] = "Metal"

    pml_layers: int = 12
    include_dispersion: bool = True
    material_fit_tolerance: float = 0.001
    material_name_to_lumerical: dict[str, str] = material_name_to_lumerical_default

    # Bent waveguide simulations
    is_bent_waveguide: bool = False
    bend_radius: float = 0
    bend_location: Literal["simulation center", "user specified"] = "user specified"
    bend_location_x: float = 0
    bend_location_y: float = 0
    bend_location_z: float = 0

    efield_intensity_threshold: float = 1e-6

    class Config:
        arbitrary_types_allowed = True

LUMERICAL_MODE_SIMULATION_SETTINGS = SimulationSettingsLumericalMode()

class SimulationSettingsLumericalInterconnect(BaseModel):
    """
    Lumerical INTERCONNECT simulation settings.

    Parameters:
        wavl_start: Start wavelength (um)
        wavl_end: End wavelength (um)
        wavl_pts: Number of wavelength points
    """
    wavl_start: float = 1.5
    wavl_end: float = 1.6
    wavl_pts: int = 1e5

    class Config:
        arbitrary_types_allowed = True

LUMERICAL_INTERCONNECT_SIMULATION_SETTINGS = SimulationSettingsLumericalInterconnect()