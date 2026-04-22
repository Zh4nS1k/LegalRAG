"""Compatibility shim for running inside the ai_service directory.

When the working directory is `ai_service/`, imports like
`ai_service.api.api` normally fail because Python looks for a nested
`ai_service/ai_service` package. Defining `__path__` turns this module into a
package rooted at the current directory, so existing absolute imports work.
"""

from __future__ import annotations

import os

__path__ = [os.path.dirname(__file__)]
