from __future__ import annotations

import pytest

from srm.config import tiny
from srm.config.build_config import BuildConfig
from srm.interface import InterfaceLayer


@pytest.fixture(scope="session")
def cfg() -> BuildConfig:
    return tiny()


@pytest.fixture()
def interface(cfg: BuildConfig) -> InterfaceLayer:
    return InterfaceLayer(cfg)
