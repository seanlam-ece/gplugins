from pathlib import Path
from gplugins.lumerical.recipes.taper_design_recipe import (
    RoutingTaperDesignRecipe,
    RoutingTaperDesignIntent,
    RoutingTaperEmeDesignRecipe,
)
from gplugins.lumerical.recipes.waveguide_recipe import (
    WaveguideSweepRecipe,
    WaveguideSweepDesignIntent,
    grillot_strip_waveguide_loss_model,
)
from functools import partial
import gdsfactory as gf
from gplugins.lumerical.simulation_settings import (
    SimulationSettingsLumericalEme,
    SimulationSettingsLumericalFdtd,
    SimulationSettingsLumericalMode,
)
from gplugins.lumerical.convergence_settings import (
    ConvergenceSettingsLumericalEme,
    ConvergenceSettingsLumericalFdtd,
    ConvergenceSettingsLumericalMode,
)
from gdsfactory.config import logger
from gdsfactory.components.taper_cross_section import taper_cross_section
from gdsfactory.components.straight import straight

from gplugins.lumerical.config import marker_list
from gplugins.design_recipe.DesignRecipe import Results
from gplugins.lumerical.config import um, cm, nm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Qt5Agg')
plt.rcParams.update({'font.size': 16})

### 0. DEFINE WHERE FILES ARE SAVED
dirpath = Path("../recipes/recipe_runs")
dirpath.mkdir(parents=True, exist_ok=True)

### 1. DEFINE DESIGN INTENT
design_intent = RoutingTaperDesignIntent(
    narrow_waveguide_routing_loss_per_cm=3,  # dB/cm
)

narrow_waveguide_cross_section = partial(
    gf.cross_section.cross_section,
    layer=(1, 0),
    width=0.5,
)
wide_waveguide_cross_section = partial(
    gf.cross_section.cross_section,
    layer=(1, 0),
    width=3.0,
)

# Waveguide loss model
wg_design_intent = WaveguideSweepDesignIntent(
    param_name="width",
    param_vals=list(np.linspace(0.5, 3.0, 26))
)

### 2. DEFINE LAYER STACK
from gdsfactory.technology.layer_stack import LayerLevel, LayerStack

layerstack_lumerical = LayerStack(
    layers={
        "box": LayerLevel(
            layer=(99999, 0),
            thickness=3.0,
            zmin=-3.0,
            material="sio2",
            sidewall_angle=0.0,
            mesh_order=9,
            layer_type="background",
        ),
        "core": LayerLevel(
            layer=(1, 0),
            thickness=0.22,
            zmin=0.0,
            material="si",
            sidewall_angle=2.0,
            width_to_z=0.5,
            mesh_order=2,
            layer_type="grow",
            info={"active": True},
        ),
    }
)

### 3. DEFINE SIMULATION AND CONVERGENCE SETTINGS
eme_convergence_setup = ConvergenceSettingsLumericalEme(
    sparam_diff=1 - 10 ** (-0.005 / 10)
)
eme_simulation_setup = SimulationSettingsLumericalEme()

fdtd_convergence_setup = ConvergenceSettingsLumericalFdtd(
    sparam_diff=0.01
)
fdtd_simulation_setup = SimulationSettingsLumericalFdtd(
    mesh_accuracy=4, port_translation=1.0, port_field_intensity_threshold=1e-5,
)

mode_convergence_setup = ConvergenceSettingsLumericalMode(neff_diff=5e-3,
                                                     ng_diff=5e-3)
mode_simulation_setup = SimulationSettingsLumericalMode(injection_axis="2D X normal",
                                                   mesh_cells_per_wavl=80,
                                                   num_modes=10,
                                                   target_mode=1,
                                                   )

wg_recipe = WaveguideSweepRecipe(cell=straight,
                                    design_intent=wg_design_intent,
                                    simulation_setup=mode_simulation_setup,
                                    convergence_setup=mode_convergence_setup,
                                    layer_stack=layerstack_lumerical,
                                    dirpath=dirpath)
wg_recipe.override_recipe = False
success = wg_recipe.eval()
if success:
    logger.info("Completed waveguide recipe.")
else:
    logger.info("Incomplete run of waveguide recipe.")

# Get waveguide loss model
loss_model = grillot_strip_waveguide_loss_model(neff_vs_width=wg_recipe.recipe_results.waveguide_sweep_characteristics,
                                                wavelength=mode_simulation_setup.wavl,
                                                )
eme_recipe = RoutingTaperEmeDesignRecipe(
    cell=taper_cross_section,
    cross_section1=narrow_waveguide_cross_section,
    cross_section2=wide_waveguide_cross_section,
    design_intent=design_intent,
    waveguide_loss_model=loss_model,
    layer_stack=layerstack_lumerical,
    simulation_setup=eme_simulation_setup,
    convergence_setup=eme_convergence_setup,
    dirpath=dirpath,
)
eme_recipe.override_recipe = False
success = eme_recipe.eval()

taper_recipe = RoutingTaperDesignRecipe(cell=taper_cross_section,
                                        cross_section1=narrow_waveguide_cross_section,
                                        cross_section2=wide_waveguide_cross_section,
                                        design_intent=design_intent,
                                        # waveguide_loss_model=loss_model,
                                        eme_simulation_setup=eme_simulation_setup,
                                        eme_convergence_setup=eme_convergence_setup,
                                        fdtd_simulation_setup=fdtd_simulation_setup,
                                        fdtd_convergence_setup=fdtd_convergence_setup,
                                        layer_stack=layerstack_lumerical,
                                        dirpath=dirpath)
taper_recipe.override_recipe = False
success = taper_recipe.eval()



# Define paths and files
experimental_result_dir = Path("./experimental_results")
experimental_result_files = ["taper_sin_length_sweep.csv",
                             "taper_lin_length_sweep.csv",
                             "taper_par_length_sweep.csv"]

convergence_result_dirs = ["../recipes/recipe_runs/LumericalEmeSimulation_1659413800206388860",
               "../recipes/recipe_runs/LumericalEmeSimulation_1900041402257800723",
               "../recipes/recipe_runs/LumericalEmeSimulation_1974935835576324923",
               "../recipes/recipe_runs/LumericalFdtdSimulation_783168376414286112",
               "../recipes/recipe_runs/LumericalFdtdSimulation_1831092248094548154",
               "../recipes/recipe_runs/LumericalFdtdSimulation_849377665587599662",
               ]

recipe_results_dirs = ["../recipes/recipe_runs/FdtdRecipe_1099282917928882571357976049617852443618803111406",
                       "../recipes/recipe_runs/FdtdRecipe_841385152542376598324042425339968224667292078441",
                       "../recipes/recipe_runs/FdtdRecipe_445410350270759193246140752037598286282350426356",
                       ]

geometry = ["Sine",
            "Linear",
            "Parabolic"]

convergence_results = []
for rdir in convergence_result_dirs:
    results = Results(prefix="convergence", dirpath=Path(rdir).resolve())
    convergence_results.append(results.get_pickle())

recipe_results = []
for rdir in recipe_results_dirs:
    results = Results(prefix="recipe", dirpath=Path(rdir).resolve())
    recipe_results.append(results.get_pickle())

for i in range(0, len(convergence_results)):
    cr = convergence_results[i]

    if "Eme" in convergence_result_dirs[i]:
        plt.figure()
        mesh_cells_per_wavl = cr.overall_convergence_data.mesh_cells
        num_cells = cr.overall_convergence_data.num_cells

        mesh_cells_per_wavl = mesh_cells_per_wavl[:-1]
        num_cells = num_cells[:-1]

        s21 = cr.overall_convergence_data.s21
        s11 = cr.overall_convergence_data.s11

        s21 = np.array(s21)
        s11 = np.array(s11)

        s21 = abs(s21[-1] - s21)
        s11 = abs(s11[-1] - s11)
        s21 = s21[:-1]
        s11 = s11[:-1]

        plt.plot(list(range(0, len(mesh_cells_per_wavl))), s21, marker=marker_list[0])
        plt.plot(list(range(0, len(mesh_cells_per_wavl))), s11, marker=marker_list[1])
        plt.xticks(
            list(range(0, len(mesh_cells_per_wavl))),
            [
                f"{mesh_cells_per_wavl[i]} | {num_cells[i][0]}"
                for i in range(0, len(mesh_cells_per_wavl))
            ],
        )
        plt.setp(plt.xticks()[1], rotation=75, horizontalalignment="center")
        plt.legend(["|S21|^2", "|S11|^2"])
        plt.grid("on")
        plt.xlabel("Mesh Cells Per Wavelength | Num Cells")
        plt.ylabel("S-Param Variation")
        plt.title(f"Mesh and Cell Convergence")
        plt.tight_layout()

        # Plot mode convergence
        plt.figure()
        modes = cr.mode_convergence_data.modes
        modes = modes[:-1]

        s21 = np.array(cr.mode_convergence_data.s21)
        s11 = np.array(cr.mode_convergence_data.s11)

        s21 = abs(s21[-1] - s21)
        s11 = abs(s11[-1] - s11)
        s21 = s21[:-1]
        s11 = s11[:-1]

        plt.plot(modes, s21, marker=marker_list[0])
        plt.plot(modes, s11, marker=marker_list[1])
        plt.legend(["|S21|^2", "|S11|^2"])
        plt.grid("on")
        plt.xlabel("Number of Modes")
        plt.ylabel("S-Param Variation")
        plt.title(f"Mode Convergence")
        plt.tight_layout()
    elif "Fdtd" in convergence_result_dirs[i]:
        # Mesh convergence
        plt.figure()
        mesh = cr.mesh_convergence_data.mesh_accuracy
        mesh = mesh[:-1]

        s21 = np.array([v[0] for v in cr.mesh_convergence_data.S21])
        s11 = np.array([v[0] for v in cr.mesh_convergence_data.S11])
        s12 = np.array([v[0] for v in cr.mesh_convergence_data.S12])
        s22 = np.array([v[0] for v in cr.mesh_convergence_data.S22])

        s21 = abs(s21[-1] - s21)
        s11 = abs(s11[-1] - s11)
        s12 = abs(s12[-1] - s12)
        s22 = abs(s22[-1] - s22)

        s21 = s21[:-1]
        s11 = s11[:-1]
        s22 = s22[:-1]
        s12 = s12[:-1]

        plt.plot(mesh, s11, marker=marker_list[0])
        plt.plot(mesh, s12, marker=marker_list[1])
        plt.plot(mesh, s21, marker=marker_list[2])
        plt.plot(mesh, s22, marker=marker_list[3])
        plt.legend(["|S11|^2", "|S12|^2", "|S21|^2", "|S22|^2"])
        plt.grid("on")
        plt.xlabel("Mesh Accuracy")
        plt.ylabel("S-Param Variation")
        plt.title(f"Mesh Convergence")
        plt.tight_layout()

        # Field intensity convergence
        plt.figure()
        thresholds = cr.field_intensity_convergence_data.thresholds
        thresholds = thresholds[:-1]

        s21 = np.array([v[0] for v in cr.field_intensity_convergence_data.S21])
        s11 = np.array([v[0] for v in cr.field_intensity_convergence_data.S11])
        s12 = np.array([v[0] for v in cr.field_intensity_convergence_data.S12])
        s22 = np.array([v[0] for v in cr.field_intensity_convergence_data.S22])

        s21 = abs(s21[-1] - s21)
        s11 = abs(s11[-1] - s11)
        s12 = abs(s12[-1] - s12)
        s22 = abs(s22[-1] - s22)

        s21 = s21[:-1]
        s11 = s11[:-1]
        s22 = s22[:-1]
        s12 = s12[:-1]

        plt.plot(thresholds, s11, marker=marker_list[0])
        plt.plot(thresholds, s12, marker=marker_list[1])
        plt.plot(thresholds, s21, marker=marker_list[2])
        plt.plot(thresholds, s22, marker=marker_list[3])
        ax = plt.gca()
        ax.invert_xaxis()
        plt.xscale("log")
        plt.legend(["|S11|^2", "|S12|^2", "|S21|^2", "|S22|^2"])
        plt.grid("on")
        plt.xlabel("Electric Field Intensity (V/m)^2")
        plt.ylabel("S-Param Variation")
        plt.title(f"Port Field Intensity Convergence")
        plt.tight_layout()

# Broadband spectrum
fig, (ax1, ax2) = plt.subplots(2)
for i in range(0, len(recipe_results)):
    wavl = recipe_results[i].sparameters.wavelength * um / nm



    s21 = 10 * np.log10( abs(recipe_results[i].sparameters.S21) ** 2)
    s11 = 10 * np.log10(abs(recipe_results[i].sparameters.S11) ** 2)

    ax1.plot(wavl, s21, label=geometry[i], marker=marker_list[i])
    ax2.plot(wavl, s11, label=geometry[i], marker=marker_list[i])

    ax2.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Transmission |S21|^2 (dB)")
    ax2.set_ylabel("Reflection |S11|^2 (dB)")

    ax1.grid("on")
    ax2.grid("on")
    ax1.legend(loc="lower right")
    ax2.legend(loc="lower right")

ax1.set_title("Taper S-Parameters")

# Plot simulated length sweeps
plt.figure()
for i in range(0, len(eme_recipe.recipe_results.length_sweeps)):
    plt.plot(eme_recipe.recipe_results.length_sweeps[i]["length"]*1e6,
             10 * np.log10(abs(eme_recipe.recipe_results.length_sweeps[i]["s21"]) ** 2),
             marker=marker_list[i],
             label=eme_recipe.recipe_results.components_settings[i]['width_type'][0].upper()+eme_recipe.recipe_results.components_settings[i]['width_type'][1:])
pos = np.linspace(0, 200, 201)
plt.plot(pos, -pos * um / cm * design_intent.narrow_waveguide_routing_loss_per_cm,
         marker=marker_list[i+1], label=f"Propagation Loss {design_intent.narrow_waveguide_routing_loss_per_cm}dB/cm")
plt.xlabel("Length (um)")
plt.ylabel("Insertion Loss (dB)")
plt.title(f"Length Sweep - Simulation Results")
plt.grid("on")
plt.legend(loc="lower center")
plt.tight_layout()

# Plot experimental length sweeps
plt.figure()
for i in range(0, len(experimental_result_files)):
    exp_file = experimental_result_dir / experimental_result_files[i]
    df = pd.read_csv(str(exp_file.resolve()), index_col=0)

    plt.plot(df.loc[:, "length"],
             df.loc[:, "loss"],
             marker=marker_list[i],
             label=geometry[i])
pos = np.linspace(0, 100, 101)
plt.plot(pos,
         -pos * um / cm * design_intent.narrow_waveguide_routing_loss_per_cm,
         marker=marker_list[i + 1],
         label=f"Propagation Loss {design_intent.narrow_waveguide_routing_loss_per_cm}dB/cm")
plt.xlabel("Length (um)")
plt.ylabel("Insertion Loss (dB)")
plt.title(f"Length Sweep - Experimental Results")
plt.grid("on")
plt.legend(loc="lower center")
plt.tight_layout()


# Plot mode overlaps
for i in range(0, len(eme_recipe.recipe_results.mode_coupling)):
    plt.figure()
    for col in eme_recipe.recipe_results.mode_coupling[i]["overlap"].columns:
        plt.plot(eme_recipe.recipe_results.mode_coupling[i]["position"],
                 eme_recipe.recipe_results.mode_coupling[i]["overlap"][col],
                 marker=marker_list[i],
                 label=col)
    plt.xlabel("Position (um)")
    plt.ylabel("Coupling (dB)")
    plt.title(f"Overlap Coupling Coefficient - {eme_recipe.recipe_results.components_settings[i]['width_type']}")
    plt.grid("on")
    plt.legend()

# Get propagation loss
from gdsfactory.typings import WidthTypes
import typing
di = design_intent

eme_device_length = 5
components = [
    taper_cross_section(
        cross_section1=narrow_waveguide_cross_section,
        cross_section2=wide_waveguide_cross_section,
        length=eme_device_length,  # um
        width_type=wtype,
    )
    for wtype in typing.get_args(WidthTypes)
]
geometries = [wtype for wtype in typing.get_args(WidthTypes)]
length_sweep_prop_loss = []
for component in components:
    propagation_loss = [0] * di.num_pts
    if hasattr(eme_recipe.recipe_setup, "waveguide_loss_model"):
        wlm = eme_recipe.recipe_setup.waveguide_loss_model

        pts = component.polygons[0].points
        # Iterate in coordinates from input of taper to end of taper and average the width between two points
        # Then, calculate loss for normalized length of device
        normalized_loss = 0
        for i in range(0, np.argmin(np.abs(pts[:, 0] - eme_device_length))):
            width = np.abs(pts[i, 1] + pts[i + 1, 1])
            length = np.abs(pts[i, 0] - pts[i + 1, 0]) * um / cm

            loss_db_per_cm = np.interp(width, wlm.loc[:, "width"],
                                       wlm.loc[:, "loss"])
            normalized_loss += loss_db_per_cm * length

        length_sweep_prop_loss.append(
            normalized_loss / eme_device_length * np.linspace(di.start_length,
                                                              di.stop_length,
                                                              di.num_pts))

# Compare simulation and experimental length sweeps
eme_recipe.recipe_results.length_sweeps.pop(0) # Remove elliptical taper
eme_recipe.recipe_results.components_settings.pop(0) # Remove elliptical taper
length_sweep_prop_loss.pop(0) # Remove elliptical taper

for i in range(0, len(eme_recipe.recipe_results.length_sweeps)):
    plt.figure()

    # Plot simulated taper
    plt.plot(eme_recipe.recipe_results.length_sweeps[i]["length"]*1e6,
             10 * np.log10(abs(eme_recipe.recipe_results.length_sweeps[i]["s21"]) ** 2),
             label="Simulation - Total Loss")

    plt.plot(eme_recipe.recipe_results.length_sweeps[i]["length"] * 1e6,
             -length_sweep_prop_loss[i],
             label="Simulation - Taper Propagation Loss")

    plt.plot(eme_recipe.recipe_results.length_sweeps[i]["length"] * 1e6,
             10 * np.log10(abs(eme_recipe.recipe_results.length_sweeps[i]["s21"]) ** 2) + length_sweep_prop_loss[i],
             label="Simulation - Taper Reflection\n+ Modal Crosstalk Loss")

    # Plot experimental taper
    exp_file = experimental_result_dir / experimental_result_files[i]
    df = pd.read_csv(str(exp_file.resolve()), index_col=0)

    plt.plot(df.loc[:, "length"],
             df.loc[:, "loss"],
             marker=marker_list[i],
             label="Experiment")

    # Plot waveguide propagation loss
    pos = np.linspace(0, 200, 201)
    plt.plot(pos, -pos * um / cm * design_intent.narrow_waveguide_routing_loss_per_cm,
             label=f"Narrow Waveguide\nPropagation Loss {design_intent.narrow_waveguide_routing_loss_per_cm}dB/cm")

    plt.xlabel("Length (um)")
    plt.ylabel("Insertion Loss (dB)")
    plt.title(f"Length Sweep - {geometry[i]}")
    plt.grid("on")
    plt.xlim([0, 100])
    plt.ylim([-1.8, 0.1 ])
    plt.legend(loc="lower right")
    plt.tight_layout()

plt.show()


if success:
    logger.info("Completed taper design recipe.")
else:
    logger.info("Incomplete run of taper design recipe.")