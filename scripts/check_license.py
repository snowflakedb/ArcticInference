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

import subprocess
import sys
from pathlib import Path
from typing import List

LICENSE_TEXT = """\
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
"""


def file_has_license(file_path: Path) -> bool:
    with open(file_path, "r") as f:
        content = f.read()
        return content.startswith(LICENSE_TEXT)


def add_license_to_file(file_path: Path) -> None:
    with open(file_path, "r+") as f:
        content = f.read()
        f.seek(0)
        f.write(LICENSE_TEXT + "\n" + content)


def get_tracked_python_files() -> List[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "--", "*.py"],
            check=True,
            text=True,
            capture_output=True,
        )
        return [Path(file) for file in result.stdout.strip().split("\n") if file]
    except subprocess.CalledProcessError as e:
        print("Error while fetching tracked files:", e, file=sys.stderr)
        sys.exit(1)


def main() -> None:
    tracked_files = get_tracked_python_files()
    modified_files = []

    for file_path in tracked_files:
        if not file_has_license(file_path):
            add_license_to_file(file_path)
            modified_files.append(file_path)

    if modified_files:
        print("The following files were updated to include the license:")
        for file in modified_files:
            print(f"  - {file}")
        sys.exit(1)  # Exit with non-zero status to fail the pre-commit hook


if __name__ == "__main__":
    main()
