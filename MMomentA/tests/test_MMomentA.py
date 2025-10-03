"""
Unit and regression test for the MMomentA package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import MMomentA


def test_MMomentA_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "MMomentA" in sys.modules
