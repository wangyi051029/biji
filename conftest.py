# -*- coding: utf-8 -*-
"""pytest 根级 conftest：确保 backend 包可从仓库根导入。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
