"""Tests for asc.__main__ entry point."""

import importlib
import runpy
import sys
from unittest.mock import patch


class TestMainEntryCallsMain:
    """Test that importing asc.__main__ calls main() from asc.cli.main."""

    def test_import_calls_main(self) -> None:
        """Importing asc.__main__ should invoke asc.cli.main.main()."""
        with patch("asc.cli.main.main") as mock_main:
            # Remove cached module so re-import triggers the top-level call
            sys.modules.pop("asc.__main__", None)

            import asc.__main__  # noqa: F401

            mock_main.assert_called_once()

    def test_reload_calls_main_again(self) -> None:
        """Reloading asc.__main__ should call main() again."""
        with patch("asc.cli.main.main") as mock_main:
            sys.modules.pop("asc.__main__", None)

            import asc.__main__  # noqa: F401

            assert mock_main.call_count == 1

            importlib.reload(asc.__main__)
            assert mock_main.call_count == 2


class TestMainEntryRunModule:
    """Test that python -m asc works via runpy."""

    def test_run_module_calls_main(self) -> None:
        """runpy.run_module('asc') should invoke asc.cli.main.main()."""
        with patch("asc.cli.main.main") as mock_main:
            # runpy may cache; clear to ensure fresh execution
            sys.modules.pop("asc.__main__", None)

            runpy.run_module("asc", run_name="__main__")

            mock_main.assert_called_once()

    def test_run_module_exit_code_zero(self) -> None:
        """python -m asc should exit cleanly when main() succeeds."""
        with patch("asc.cli.main.main"):
            sys.modules.pop("asc.__main__", None)

            result = runpy.run_module("asc", run_name="__main__")

            # runpy returns the module dict; no exception means success
            assert isinstance(result, dict)
