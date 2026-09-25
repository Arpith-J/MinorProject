"""Root pytest configuration.

Ensures the project root is on ``sys.path`` so that ``import src`` works
from anywhere inside the repository (pytest inserts the directory that
contains this file into ``sys.path``).
"""