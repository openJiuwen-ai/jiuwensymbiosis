# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Developer scripts package.

Making ``scripts`` a real package (vs. an implicit namespace package) lets
mypy resolve ``scripts.<name>`` unambiguously — without this, a file passed to
mypy both directly (``scripts/foo.py``) and via ``from scripts.foo import ...``
is seen under two module names. Sub-script dirs that are meant to be packages
(``new_adapter/``) already carry their own ``__init__.py``.
"""
