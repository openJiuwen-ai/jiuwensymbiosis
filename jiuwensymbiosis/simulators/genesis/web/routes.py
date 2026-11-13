"""Compatibility import for shared simulation routes."""

from jiuwensymbiosis.simulators.common.web.routes import install_simulation_routes

install_genesis_routes = install_simulation_routes
__all__ = ["install_genesis_routes"]
