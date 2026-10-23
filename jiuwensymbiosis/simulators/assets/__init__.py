"""本仓库独立的本地资产目录；构建和查询不需要仿真 SDK。"""

from .companions import validate_material_bundle
from .config import AssetConfig, initialize_config, load_config
from .index import AssetLibrary, build_index
from .models import Asset, AssetError, AssetPage, ResolvedAsset
from .probe import probe_asset

__all__ = [
    "Asset",
    "AssetConfig",
    "AssetError",
    "AssetLibrary",
    "AssetPage",
    "ResolvedAsset",
    "build_index",
    "load_config",
    "initialize_config",
    "probe_asset",
    "validate_material_bundle",
]
