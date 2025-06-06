# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import subprocess
import sys
from typing import Dict
from typing import List

try:
    import tomllib
except (ImportError, ModuleNotFoundError):
    from pip._vendor import tomli as tomllib

PYPROJECT_FILE = "pyproject.toml"


def load_optional_dependencies() -> Dict[str, List[str]]:
    """Load optional dependencies from pyproject.toml using tomllib."""
    with open(PYPROJECT_FILE, "rb") as f:  # Open file in binary mode
        pyproject = tomllib.load(f)
    return pyproject.get("project", {}).get("optional-dependencies", {})


def resolve_dependencies(groups: List[str], optional_deps: Dict[str, List[str]]) -> List[str]:
    """Resolve dependencies, including references to other groups."""
    resolved = set()
    stack = list(groups)  # Stack to process groups

    while stack:
        group = stack.pop()
        if group not in optional_deps:
            raise ValueError(
                f"Group '{group}' does not exist in optional dependencies. Optional groups: {', '.join(optional_deps)}"
            )

        for dep in optional_deps[group]:
            if dep.startswith("arctic_training["):  # Handle group references
                ref_group = dep.split("[")[1].rstrip("]")
                stack.append(ref_group)
            else:
                resolved.add(dep)

    return sorted(resolved)  # Return sorted list for consistency


def install_dependencies(groups: List[str]) -> None:
    """Install dependencies for the specified groups."""
    optional_deps = load_optional_dependencies()
    all_deps = resolve_dependencies(groups, optional_deps)

    if not all_deps:
        print("No dependencies to install.")
        return

    subprocess.check_call([sys.executable, "-m", "pip", "install"] + all_deps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install optional dependencies from pyproject.toml.")
    parser.add_argument(
        "groups",
        nargs="+",
        help="The names of the optional dependency groups to install (e.g., docs, dev).",
    )
    args = parser.parse_args()
    install_dependencies(args.groups)


if __name__ == "__main__":
    main()