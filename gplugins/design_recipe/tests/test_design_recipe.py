from gplugins.design_recipe.DesignRecipe import DesignRecipe, eval_decorator
from gplugins.design_recipe.ConstituentRecipes import ConstituentRecipes
from gdsfactory import Component
from gdsfactory.components.straight import straight
from gdsfactory import logger
from gdsfactory.pdk import LayerStack, get_layer_stack
from gdsfactory.technology.processes import Etch, Grow
from pathlib import Path
from gplugins.lumerical.convergence_settings import LUMERICAL_FDTD_CONVERGENCE_SETTINGS

def test_design_recipe():
    class CustomRecipe1(DesignRecipe):
        def __init__(self,
                     component: Component | None = None,
                     layer_stack: LayerStack | None = None,
                     process: tuple | None = None,
                     dependencies: list[DesignRecipe] | None = None,
                     dirpath: Path | None = None,
                     ):
            super().__init__(cell=component, dependencies=dependencies, layer_stack=layer_stack, process=process, dirpath=dirpath)
            self.recipe_setup.test_setup1 = {"a": [1,2,3], "b": [4,3,2]}
            self.recipe_setup.test_setup2 = [1, 2, 3]

        @eval_decorator
        def eval(self, run_convergence: bool = True) -> bool:
            self.recipe_results.results1 = [5,4,3,2,1]
            return True

    class CustomRecipe2(DesignRecipe):
        def __init__(self,
                     component: Component | None = None,
                     layer_stack: LayerStack | None = None,
                     process: tuple | None = None,
                     dependencies: list[DesignRecipe] | None = None,
                     dirpath: Path | None = None,
                     ):
            super().__init__(cell=component, dependencies=dependencies, layer_stack=layer_stack, process=process, dirpath=dirpath)
            self.recipe_setup.test_setup1 = "abcdefg"
            self.recipe_setup.test_setup2 = LUMERICAL_FDTD_CONVERGENCE_SETTINGS

        @eval_decorator
        def eval(self, run_convergence: bool = True) -> bool:
            self.recipe_results.results1 = 42
            return True

    class CustomRecipe3(DesignRecipe):
        def __init__(self,
                     component: Component | None = None,
                     layer_stack: LayerStack | None = None,
                     process: tuple | None = None,
                     dependencies: list[DesignRecipe] | None = None,
                     dirpath: Path | None = None,
                     ):
            super().__init__(cell=component, dependencies=dependencies, layer_stack=layer_stack, process=process, dirpath=dirpath)
            self.recipe_setup.test_setup1 = "testing"
            self.recipe_setup.test_setup2 = (1+1j)

        @eval_decorator
        def eval(self, run_convergence: bool = True) -> bool:
            self.recipe_results.results1 = (1+2j)
            return True

    # Specify test directory
    dirpath = Path(__file__).resolve().parent / "test_recipe"
    dirpath.mkdir(parents=True, exist_ok=True)

    # Clean up test directory if data is there
    import shutil
    check_dirpath1 = dirpath / "CustomRecipe1_1212138028880644074211835372326109759715465388193"
    check_dirpath2 = dirpath / "CustomRecipe2_212169082254383719714582730552051364969517410496"
    check_dirpath3 = dirpath / "CustomRecipe3_780227881654101345764575896110287802288520705536"
    if check_dirpath1.is_dir():
        shutil.rmtree(str(check_dirpath1))
    if check_dirpath2.is_dir():
        shutil.rmtree(str(check_dirpath2))
    if check_dirpath3.is_dir():
        shutil.rmtree(str(check_dirpath3))

    # Set up recipes
    component1 = straight(width=0.6)
    component2 = straight(width=0.5)
    component3 = straight(width=0.4)
    A = CustomRecipe1(component=component1, layer_stack=get_layer_stack(), process=(Etch(name="test1", material="si", depth=2.0), Grow(name="test2", material="si1", type="isotropic", thickness=1.0)), dirpath=dirpath)
    B = CustomRecipe2(component=component2, layer_stack=get_layer_stack(), process=(Etch(name="test3", material="si", depth=3.0), Grow(name="test4", material="si1", type="isotropic", thickness=1.0)), dirpath=dirpath)
    C = CustomRecipe3(component=component3, layer_stack=get_layer_stack(), process=(Etch(name="test5", material="si", depth=4.0), Grow(name="test6", material="si1", type="isotropic", thickness=1.0)), dirpath=dirpath)

    # Test with no dependencies
    A.eval()

    # Check if recipe directory exists
    check_dirpath = dirpath / "CustomRecipe1_1212138028880644074211835372326109759715465388193"
    if not check_dirpath.is_dir():
        raise NotADirectoryError("Recipe directory not created with hash.")

    # Check if recipe results exists
    check_results = check_dirpath / "recipe_results.pkl"
    if not check_results.is_file():
        raise FileNotFoundError("Recipe results (recipe_results.pkl) not found.")

    # Check if recipe dependencies exists
    check_dependencies = check_dirpath / "recipe_dependencies.txt"
    if not check_dependencies.is_file():
        raise FileNotFoundError("Recipe results (recipe_dependencies.txt) not found.")

    # Check if recipe dependencies has any characters or values. Empty is expected.
    with open(check_dependencies, "r") as f:
        chars = f.read()
        if len(chars) > 1:
            raise ValueError("Recipe dependencies should be empty.")

    # Test with dependencies
    A.dependencies = [B, C]
    A.eval()

    # Check if recipe dependencies has any characters or values. Empty is expected.
    with open(check_dependencies, "r") as f:
        chars = f.read()
        if not len(chars) > 1:
            raise ValueError("Recipe dependencies should have dependencies listed.")

    # Check if dependent recipes exist
    check_dirpath = dirpath / "CustomRecipe2_212169082254383719714582730552051364969517410496"
    check_results = check_dirpath / "recipe_results.pkl"
    check_dependencies = check_dirpath / "recipe_dependencies.txt"
    if not check_dirpath.is_dir():
        raise NotADirectoryError("Recipe directory not created with hash.")
    if not check_results.is_file():
        raise FileNotFoundError("Recipe results (recipe_results.pkl) not found.")
    if not check_dependencies.is_file():
        raise FileNotFoundError("Recipe results (recipe_dependencies.txt) not found.")

    check_dirpath = dirpath / "CustomRecipe3_780227881654101345764575896110287802288520705536"
    check_results = check_dirpath / "recipe_results.pkl"
    check_dependencies = check_dirpath / "recipe_dependencies.txt"
    if not check_dirpath.is_dir():
        raise NotADirectoryError("Recipe directory not created with hash.")
    if not check_results.is_file():
        raise FileNotFoundError("Recipe results (recipe_results.pkl) not found.")
    if not check_dependencies.is_file():
        raise FileNotFoundError("Recipe results (recipe_dependencies.txt) not found.")