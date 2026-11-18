"""HTTP routes, one module per part of the UI that calls them.

A new route goes into the module of the page that calls it, and does its work in the modules
it calls (srl/launch.py, srl/runs.py, srl/aws/...), not in the route. app.py only assembles;
tests/test_app.py checks that every /api route is defined here.
"""

from __future__ import annotations

from . import auth, catalog, mlflow, resources, runs, setup

ROUTERS = [auth.router, setup.router, catalog.router, runs.router, mlflow.router, resources.router]
