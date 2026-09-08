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

"""Tests for Ulysses sequence parallelism configuration validation."""

import pytest
from unittest.mock import MagicMock, patch


class TestUlyssesConfigValidation:
    """Test cases for Ulysses config validation in VllmConfigPatch."""

    def test_max_num_seqs_divisibility_error_message(self):
        """Test that non-divisible max_num_seqs raises ValueError with helpful message."""
        from arctic_inference.vllm.config import VllmConfigPatch

        # Create a mock VllmConfig instance
        mock_config = MagicMock()
        mock_config.parallel_config.ulysses_sequence_parallel_size = 4
        mock_config.scheduler_config.max_num_seqs = 37  # Not divisible by 4

        # Call the validation method directly
        with pytest.raises(ValueError) as excinfo:
            VllmConfigPatch._validate_ulysses_config(mock_config)

        error_msg = str(excinfo.value)
        # Check that error message contains helpful information
        assert "ulysses_sequence_parallel_size=4" in error_msg
        assert "--max-num-seqs (37)" in error_msg
        assert "must be divisible by" in error_msg
        # Check that it suggests valid values
        assert "--max-num-seqs=36" in error_msg or "--max-num-seqs=40" in error_msg

    def test_max_num_seqs_divisible_passes(self):
        """Test that divisible max_num_seqs passes validation."""
        from arctic_inference.vllm.config import VllmConfigPatch

        mock_config = MagicMock()
        mock_config.parallel_config.ulysses_sequence_parallel_size = 4
        mock_config.scheduler_config.max_num_seqs = 36  # Divisible by 4

        # Should not raise
        VllmConfigPatch._validate_ulysses_config(mock_config)

    def test_sp_size_one_skips_validation(self):
        """Test that validation is skipped when sp_size is 1."""
        from arctic_inference.vllm.config import VllmConfigPatch

        mock_config = MagicMock()
        mock_config.parallel_config.ulysses_sequence_parallel_size = 1
        mock_config.scheduler_config.max_num_seqs = 37  # Any value should work

        # Should not raise
        VllmConfigPatch._validate_ulysses_config(mock_config)

    def test_various_non_divisible_values(self):
        """Test various non-divisible max_num_seqs values."""
        from arctic_inference.vllm.config import VllmConfigPatch

        test_cases = [
            (4, 37),   # 37 % 4 = 1
            (4, 38),   # 38 % 4 = 2
            (4, 39),   # 39 % 4 = 3
            (8, 100),  # 100 % 8 = 4
            (2, 33),   # 33 % 2 = 1
        ]

        for sp_size, max_num_seqs in test_cases:
            mock_config = MagicMock()
            mock_config.parallel_config.ulysses_sequence_parallel_size = sp_size
            mock_config.scheduler_config.max_num_seqs = max_num_seqs

            with pytest.raises(ValueError):
                VllmConfigPatch._validate_ulysses_config(mock_config)

    def test_various_divisible_values(self):
        """Test various divisible max_num_seqs values."""
        from arctic_inference.vllm.config import VllmConfigPatch

        test_cases = [
            (4, 36),   # 36 % 4 = 0
            (4, 40),   # 40 % 4 = 0
            (8, 96),   # 96 % 8 = 0
            (2, 32),   # 32 % 2 = 0
            (16, 64),  # 64 % 16 = 0
        ]

        for sp_size, max_num_seqs in test_cases:
            mock_config = MagicMock()
            mock_config.parallel_config.ulysses_sequence_parallel_size = sp_size
            mock_config.scheduler_config.max_num_seqs = max_num_seqs

            # Should not raise
            VllmConfigPatch._validate_ulysses_config(mock_config)
