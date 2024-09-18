import gdsfactory as gf
from gplugins.lumerical.eme import LumericalEmeSimulation
from functools import partial
from gdsfactory.components.taper_cross_section import taper_cross_section
from gdsfactory.technology.layer_stack import LayerLevel, LayerStack, LogicalLayer

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

layerstack_lumerical = LayerStack(
    layers={
        "clad": LayerLevel(
            layer=LogicalLayer(layer=(600, 0)),
            thickness=3.0,
            zmin=0.0,
            material="sio2",
            sidewall_angle=0.0,
            mesh_order=9,
            info={"layer_type": "background"},
        ),
        "box": LayerLevel(
            layer=LogicalLayer(layer=(600, 0)),
            thickness=3.0,
            zmin=-3.0,
            material="sio2",
            sidewall_angle=0.0,
            mesh_order=9,
            info={"layer_type": "background"},
        ),
        "core": LayerLevel(
            layer=LogicalLayer(layer=(1, 0)),
            thickness=0.22,
            zmin=0.0,
            material="si",
            sidewall_angle=2.0,
            width_to_z=0.5,
            mesh_order=2,
            info={"active": True,
                  "layer_type": "grow"},
        ),
    }
)

sim = LumericalEmeSimulation(
    taper,
    layerstack=layerstack_lumerical,
    run_mesh_convergence=False,
    run_cell_convergence=False,
    run_overall_convergence=False,
    run_mode_convergence=False,
    hide=False,
)

data = sim.get_length_sweep()
print(data)