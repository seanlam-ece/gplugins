from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from gdsfactory import Component
from gdsfactory.path import hashlib
from gdsfactory.pdk import LayerStack, get_layer_stack
from gdsfactory.typings import ComponentFactory
from gdsfactory import logger, get_layer
from pydantic import BaseModel

import gplugins.design_recipe as dr
from gplugins.lumerical.utils import Results


class Setup:
    """
    A dynamic class to store any setup information

    This class allows designers to arbitrarily add setup information.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __eq__(self, other):
        if not isinstance(other, Setup):
            # Don't attempt to compare against unrelated types
            return NotImplemented
        return self.__dict__ == other.__dict__


class DesignRecipe:
    """
    A DesignRecipe represents a flow of operations on GDSFactory components,
    with zero or more dependencies. Note that dependencies are assumed to be independent,
    dependent dependencies should be nested. When `eval()`ed, A DesignRecipe `eval()`s
    its dependencies if they they've become stale,
    and optionally executes some tool-specific functionality.
    For example,an FdtdDesignRecipe might simulate its `component` in
    Lumerial FDTD to extract its s-parameters.


    Attributes:
        dependencies: This `DesignRecipe`s dependencies. These are assumed to be independent
        last_hash: The hash of the system last time eval() was executed. This is used to determine whether the recipe's
                    configuration has changed, meaning it must be re-eval'ed.
        cell: GDSFactory layout cell or component. This is not necessarily the same `component` referred to in the
                `dependencies` recipes.
        layer_stack: PDK layerstack
        dirpath: Root directory where recipes runs are stored
        recipe_dirpath: Recipe directory where recipe results are stored. This is only created upon eval of the recipe.
        run_convergence: Run convergence if True. Accurate simulations come from simulations that have run convergence.
        override_recipe: Overrides recipe results if True. This runs the design recipe regardless if results are
                        available and overwrites recipe results.
        recipe_results: A dynamic object used to store recipe results and the setup to these results
    """

    dependencies: dr.ConstituentRecipes
    last_hash: int = -1
    cell: ComponentFactory | Component | None = None
    layer_stack: LayerStack | None = None
    dirpath: Path | None = None
    recipe_dirpath: Path | None = None
    run_convergence: bool = True
    override_recipe: bool = False
    recipe_setup: Setup | None = None
    recipe_results: Results | None = None

    def __init__(
        self,
        cell: ComponentFactory | Component,
        dependencies: list[dr.DesignRecipe] | None = None,
        layer_stack: LayerStack = get_layer_stack(),
        dirpath: Path | None = None,
    ):
        self.dependencies = dr.ConstituentRecipes(dependencies)
        self.dirpath = dirpath or Path(".")
        self.cell = cell

        # Initialize recipe setup
        if isinstance(self.cell, Callable):
            c = self.cell()
        elif type(self.cell) == Component:
            c = self.cell
        else:
            c = self.cell

        # Change port layers to integers before hashing
        for i in range(0, len(c.ports)):
            c.ports[i].layer = get_layer(c.ports[i].layer).value

        for i in range(0, len(c.references)):
            for j in range(0, len(c.references[i].cell.ports)):
                c.references[i].cell.ports[j].layer = get_layer(
                    c.references[i].cell.ports[j].layer)

        cell_hash = c.hash()
        self.recipe_setup = Setup(cell_hash=cell_hash,
                                  layer_stack=layer_stack)
        self.recipe_results = Results(prefix="recipe")

    def __hash__(self) -> int:
        """
        Returns a hash of all state this DesignRecipe contains.
        This is used to determine 'freshness' of a recipe (i.e. if it needs to be rerun)

        Hashed items:
        - component / cell geometry
        - layer stack
        """
        h = hashlib.sha1()
        for attr in self.recipe_setup.__dict__.values():
            if (
                isinstance(attr, int)
                or isinstance(attr, float)
                or isinstance(attr, complex)
                or isinstance(attr, str)
            ):
                h.update(str(attr).encode("utf-8"))
            elif isinstance(attr, dict) or isinstance(attr, list):
                h.update(json.dumps(attr, sort_keys=True).encode("utf-8"))
            elif isinstance(attr, BaseModel):
                h.update(attr.model_dump_json().encode("utf-8"))
            else:
                h.update(str(attr).encode("utf-8"))
        h.update(str(self.run_convergence).encode("utf-8"))
        return int.from_bytes(h.digest(), "big")

    def is_fresh(self) -> bool:
        """
        Returns if this DesignRecipe needs to be re-`eval()`ed.
        This could be either caused by this DesignRecipe's
        configuration being changed, or that of one of its dependencies.
        """
        return self.__hash__() == self.last_hash and all(
            recipe.is_fresh() for recipe in self.dependencies
        )

    def eval(self, force_rerun_all=False) -> bool:
        """
        Evaluate this DesignRecipe. This should be overridden by
        subclasses with their specific functionalities
        (e.g. run the fdtd engine).
        Here we only evaluate dependencies,
        since the generic DesignRecipe has no underlying task.

        Parameters:
            force_rerun_all: Forces design recipes to be re-evaluated
        """
        success = self.eval_dependencies(force_rerun_all=force_rerun_all)

        self.last_hash = hash(self)
        return success

    def eval_dependencies(self, force_rerun_all=False) -> bool:
        """
        Evaluate this `DesignRecipe`'s dependencies.
        Because `dependencies` are assumed to be independent,
        they can be evaluated in any order.

        Parameters:
            force_rerun_all: Forces design recipes to be re-evaluated
        """
        success = True
        for recipe in self.dependencies:
            if force_rerun_all or (not recipe.is_fresh()):
                success = success and recipe.eval()
        return success

    def load_recipe_results(self):
        """
        Loads recipe results from pickle file into class attribute
        """
        self.recipe_results = self.recipe_results.get_pickle()

    def save_recipe_results(self):
        """
        Saves recipe_results to pickle file while adding setup information.
        This includes:
        - component hash
        - layerstack
        - convergence_settings
        - simulation_settings

        This is usually done after convergence testing is completed and simulation settings are accurate and should be
        saved for future reference/recall.
        """
        self.recipe_results.recipe_setup = self.recipe_setup
        self.recipe_dirpath.resolve().mkdir(parents=True, exist_ok=True)
        self.recipe_results.save_pickle()

    def is_same_recipe_results(self) -> bool:
        """
        Returns whether recipe results' setup are the same as the current setup for the recipe.
        This is important for preventing hash collisions.
        """
        try:
            return self.recipe_results.recipe_setup == self.recipe_setup
        except AttributeError:
            return False


def eval_decorator(func):
    """
    Design recipe eval decorator

    Parameters:
        func: Design recipe eval method

    Returns:
        Design recipe eval method decorated with dependency execution and hashing
    """

    def design_recipe_eval(*args, **kwargs):
        """
        Evaluates design recipe and its dependencies then hashes the design recipe and returns successful execution
        """
        self = args[0]
        # Evaluate independent dependent recipes
        success = self.eval_dependencies()

        if "run_convergence" in kwargs:
            self.run_convergence = kwargs["run_convergence"]
        self.last_hash = self.__hash__()

        # Create directory for recipe results
        self.recipe_dirpath = self.dirpath / f"{self.__class__.__name__}_{self.last_hash}"
        self.recipe_dirpath.mkdir(parents=True, exist_ok=True)
        self.recipe_results.dirpath = self.recipe_dirpath
        self.recipe_results.prefix = "recipe"
        logger.info(f"Hashed Directory: {self.recipe_dirpath}")

        # Add text file outlining dependencies for traceability
        with open(str(self.recipe_dirpath.resolve() / "recipe_dependencies.txt"), "w") as f:
            dependencies = [f"{recipe.__class__.__name__}_{recipe.last_hash}" for recipe in self.dependencies]
            dependencies = "\n".join(dependencies)
            f.write(dependencies)


        # Check if results already available. Results must be stored in directory with the same hash.
        if (
            self.recipe_results.available()
            and not self.override_recipe
            and self.is_fresh()
        ):
            # Load results if available
            try:
                self.load_recipe_results()
            except:
                self.override_recipe = True

            if not self.is_same_recipe_results() or self.override_recipe:
                # If the recipe setup is not the same as in the results, eval the design recipe
                success = success and func(*args, **kwargs)
                self.save_recipe_results()
        else:
            # If results not available, recipe config has changed, or user wants to override recipe results,
            # eval the design recipe
            success = success and func(*args, **kwargs)
            self.save_recipe_results()

        # Return successful execution
        return success

    return design_recipe_eval
