from functools import partial

import gdsfactory as gf
from gdsfactory.components.taper_cross_section import taper_cross_section

from gplugins.lumerical.convergence_settings import LUMERICAL_FDTD_CONVERGENCE_SETTINGS
from gplugins.lumerical.fdtd import LumericalFdtdSimulation
from gplugins.lumerical.simulation_settings import SIMULATION_SETTINGS_LUMERICAL_FDTD
from gdsfactory.generic_tech.layer_stack import get_layer_stack, get_process
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.use("Qt5Agg")

xs_wg = partial(
    gf.cross_section.cross_section,
    layer=(1, 0),
    width=0.5,
)

xs_wg_wide = partial(
    gf.cross_section.cross_section,
    layer=(1, 0),
    width=2.0,
)

taper = taper_cross_section(
    cross_section1=xs_wg,
    cross_section2=xs_wg_wide,
    length=5,
    width_type="parabolic",
)

layerstack = get_layer_stack()
process = get_process()

# Remove metal and nitride layers for visualization in Lumerical
layerstack.layers.pop("nitride")
layerstack.layers.pop("via_contact")
layerstack.layers.pop("metal1")
layerstack.layers.pop("heater")
layerstack.layers.pop("via1")
layerstack.layers.pop("metal2")
layerstack.layers.pop("via2")
layerstack.layers.pop("metal3")

SIMULATION_SETTINGS_LUMERICAL_FDTD.port_translation = 1.0
SIMULATION_SETTINGS_LUMERICAL_FDTD.solver_type = "gpu"
SIMULATION_SETTINGS_LUMERICAL_FDTD.frequency_dependent_profile = False

sim = LumericalFdtdSimulation(
    component=taper,
    layerstack=layerstack,
    process=process,
    convergence_settings=LUMERICAL_FDTD_CONVERGENCE_SETTINGS,
    simulation_settings=SIMULATION_SETTINGS_LUMERICAL_FDTD,
    hide=False,
    run_port_convergence=False,
    run_mesh_convergence=False,
)

sp = sim.write_sparameters(overwrite=True)
plt.figure()
plt.plot(sp.loc[:,"wavelength"], 10 * np.log10(abs(sp.loc[:, "S21"]) ** 2), label="|S21|^2")
plt.title("Transmission |S21|^2")
plt.xlabel("Wavelength (um)")
plt.ylabel("Transmission (dB)")
plt.grid("on")

plt.figure()
plt.plot(sp.loc[:,"wavelength"], 10 * np.log10(abs(sp.loc[:, "S11"]) ** 2), label="|S11|^2")
plt.title("Reflection |S11|^2")
plt.xlabel("Wavelength (um)")
plt.ylabel("Reflection (dB)")
plt.grid("on")

