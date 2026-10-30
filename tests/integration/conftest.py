"""Integration tests may import plaid (docs/methods.md §6); skip the whole folder if it's absent."""

import importlib.util

collect_ignore_glob = [] if importlib.util.find_spec("plaid") else ["test_*.py"]
