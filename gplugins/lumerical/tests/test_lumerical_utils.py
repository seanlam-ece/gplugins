from gdsfactory import logger

from gplugins.lumerical.config import DEBUG_LUMERICAL
from gplugins.lumerical.simulation_settings import material_name_to_lumerical_default
from gplugins.lumerical.utils import to_lbr


def test_to_lbr():
    """
    Ensure process file generated from function is importable in MODE, FDTD, and CHARGE.
    """
    from gdsfactory.generic_tech.layer_stack import get_process, get_layer_stack

    # Inputs
    layer_map = material_name_to_lumerical_default

    # Create layerstack
    layer_stack = get_layer_stack()
    process = get_process()

    for i in range(2, 10):
        process[i].layers_and = [(1,0)]

    for i in range(2, 10):
        if i % 2 == 0:
            process[i].ion = "p"
        else:
            process[i].ion = "n"

    # Check process file in Lumerical MODE, FDTD, and CHARGE
    try:
        import lumapi
    except Exception as err:
        raise AssertionError(
            f"{err}\nUnable to import lumapi. Check sys.path for location to lumapi.py."
        ) from err

    message = ""
    sessions = [
        lumapi.MODE(hide=not DEBUG_LUMERICAL),
        lumapi.FDTD(hide=not DEBUG_LUMERICAL),
        lumapi.DEVICE(hide=not DEBUG_LUMERICAL),
    ]
    for s in sessions:
        success = False

        # Create passive LBR process file
        process_file_lumerical = to_lbr(
            layer_map, layerstack=layer_stack, process=process,
        )
        try:
            s.addlayerbuilder()
            s.loadprocessfile(str(process_file_lumerical))
            success = success or True
            message += f"\nSUCCESS ({type(s)})"
        except Exception as err:
            success = success or False
            message += f"\n{err}\nWARNING ({type(s)})"

        s.close()

        # If process file cannot be imported into particular Lumerical simulator, raise error
        if not success:
            raise AssertionError(f"Process file cannot be imported into {type(s)}")

    logger.info(message)
    assert success, message
