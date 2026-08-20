"""Every public class the wrapper defines has to be in ``__all__`` (and vice
versa). Static: reads python/rcdl/__init__.py, needs no board and no model.
"""

import ast
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WRAPPER = os.path.join(_REPO, "python", "rcdl", "__init__.py")


def _parse():
    with open(WRAPPER, encoding="utf-8") as f:
        return ast.parse(f.read(), WRAPPER)


def _all_names(tree):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return {elt.value for elt in node.value.elts}
    raise AssertionError("__all__ not found")


def test_public_classes_are_exported():
    tree = _parse()
    exported = _all_names(tree)
    classes = {n.name for n in tree.body if isinstance(n, ast.ClassDef) and not n.name.startswith("_")}
    missing = classes - exported
    assert not missing, f"classes defined but not in __all__: {sorted(missing)}"


def test_exports_exist():
    tree = _parse()
    exported = _all_names(tree)
    defined = {n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    imported = set()
    for n in tree.body:
        if isinstance(n, ast.ImportFrom):
            imported |= {a.asname or a.name for a in n.names}
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    defined.add(t.id)
    unknown = exported - defined - imported
    assert not unknown, f"__all__ names nothing defines: {sorted(unknown)}"
