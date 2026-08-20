"""pytest configuration for RCDL.

Two layers of tests:
  * host-runnable — pure-numpy / static checks that need neither a board nor a
    model (they still import ``rcdl`` where it exists, and skip cleanly otherwise);
  * board — end-to-end on a real ``.rknn``. Point them at a model with
    ``--model`` or ``RCDL_TEST_MODEL``; they skip when neither is given.

    PYTHONPATH=build:python pytest tests/ --model models/resnet18_rk3588.rknn
"""

import os

import pytest


def pytest_addoption(parser):
    parser.addoption("--model", default=os.environ.get("RCDL_TEST_MODEL", ""),
                     help="path to an .rknn used by the board tests")


@pytest.fixture(scope="session")
def model_path(request):
    p = request.config.getoption("--model")
    if not p:
        pytest.skip("no model: pass --model <file.rknn> or set RCDL_TEST_MODEL")
    if not os.path.isfile(p):
        pytest.skip(f"model not found: {p}")
    return p


@pytest.fixture(scope="session")
def rcdl_mod():
    return pytest.importorskip("rcdl", reason="rcdl module not importable (build on the board first)")
