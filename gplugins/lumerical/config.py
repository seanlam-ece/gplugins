import numpy as np
from pathlib import Path
# Used to display Lumerical GUIs, log status, and plot results
DEBUG_LUMERICAL = False

# Enable dopant generation in lbr process file. Set this to False if using Lumerical versions 2021 or prior (Layer Builder
# has issues with dopants for these versions of Lumerical).
ENABLE_DOPING = True

# Opacity in DEVICE
OPACITY = 0.4
# Format for colors (R, G, B, opacity) for DEVICE
MATERIAL_COLORS = [
    np.array([1, 0, 0, OPACITY]),
    np.array([1, 0.5, 0, OPACITY]),
    np.array([1, 1, 0, OPACITY]),
    np.array([0.5, 1, 0, OPACITY]),
    np.array([0, 1, 1, OPACITY]),
    np.array([0, 0.5, 1, OPACITY]),
    np.array([0, 0, 1, OPACITY]),
    np.array([0.5, 0, 1, OPACITY]),
    np.array([1, 0, 1, OPACITY]),
] * 10

dbu = 1e3
um = 1e-6
cm = 1e-2
nm = 1e-9
marker_list = [
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "s",
    "p",
    "P",
    "*",
    "h",
    "+",
    "X",
    "D",
] * 10

COMPACT_MODEL_LIBRARY_PATH = Path("./recipes/compact_model_library")
if not COMPACT_MODEL_LIBRARY_PATH.resolve().is_dir():
    COMPACT_MODEL_LIBRARY_PATH.resolve().mkdir(parents=True, exist_ok=True)