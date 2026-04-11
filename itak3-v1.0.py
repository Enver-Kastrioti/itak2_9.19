#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Legacy compatibility entrypoint for iTAK."""

import sys

from itak_cli import main

REMOVAL_TARGET = "the next breaking-change cleanup release"
DEPRECATION_MESSAGE = (
    "Warning: 'itak3-v1.0.py' is deprecated and kept only for compatibility. "
    f"Use './itak' as the primary entrypoint. Planned removal: {REMOVAL_TARGET}."
)


if __name__ == "__main__":
    print(DEPRECATION_MESSAGE, file=sys.stderr)
    main()
