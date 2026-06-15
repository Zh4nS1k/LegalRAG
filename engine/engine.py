"""Compatibility shim for running inside the engine directory.

When the working directory is `engine/`, imports like
`engine.api.api` normally fail because Python looks for a nested
`engine/engine` package. Defining `__path__` turns this module into a
package rooted at the current directory, so existing absolute imports work.
"""

from __future__ import annotations

import os

__path__ = [os.path.dirname(__file__)]
