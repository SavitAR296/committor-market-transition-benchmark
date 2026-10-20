"""Writes manuscript/tables/tab1_gaussian.tex (all manuscript tables are produced together by make_tables_pre.py)."""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).with_name("make_tables_pre.py")), run_name="__main__")
