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

import pytest

from arctic_inference.patching import ArcticPatch


class TestArcticPatch:

    def setup_method(self):

        # Define a simple class to be patched in tests
        class TargetClass:

            my_field = "original field"

            def my_method(self):
                return "original method"

            @classmethod
            def my_classmethod(cls):
                return "original classmethod"

        # Create a derived class
        class DerivedClass(TargetClass):

            def my_method(self):
                return super().my_method() + " derived"

        self.TargetClass = TargetClass
        self.DerivedClass = DerivedClass

    def test_patch_adds_new_attributes(self):

        class PatchClass(ArcticPatch[self.TargetClass]):

            new_field = "new field"

            def new_method(self):
                return "new method"

        PatchClass.apply_patch()

        instance = self.TargetClass()
        assert hasattr(self.TargetClass, "new_field")
        assert instance.new_field == "new field"
        assert hasattr(self.TargetClass, "new_method")
        assert instance.new_method() == "new method"
        assert "_arctic_patches" in self.TargetClass.__dict__
        assert "new_field" in self.TargetClass._arctic_patches
        assert "new_method" in self.TargetClass._arctic_patches
        assert self.TargetClass._arctic_patches["new_method"] is PatchClass

    def test_patch_replaces_existing_attributes(self):

        class PatchClass(ArcticPatch[self.TargetClass]):

            def my_method(self):
                return "patched"

        PatchClass.apply_patch()

        instance = self.TargetClass()
        assert instance.my_method() == "patched"
        assert self.TargetClass._arctic_patches["my_method"] is PatchClass

    def test_cannot_patch_same_attribute_twice(self):

        class FirstPatch(ArcticPatch[self.TargetClass]):

            def my_method(self):
                return "first patch"

        FirstPatch.apply_patch()

        class SecondPatch(ArcticPatch[self.TargetClass]):

            def my_method(self):
                return "second patch"

        with pytest.raises(ValueError, match="is already patched by"):
            SecondPatch.apply_patch()

    def test_patch_method_with_inheritance(self):

        class PatchClass(ArcticPatch[self.TargetClass]):

            def my_method(self):
                return self.__class__.__name__

        PatchClass.apply_patch()

        instance = self.TargetClass()
        derived = self.DerivedClass()
        assert instance.my_method() == "TargetClass"
        assert derived.my_method() == "DerivedClass derived"

    def test_patch_classmethod_with_inheritance(self):

        class PatchClass(ArcticPatch[self.TargetClass]):

            @classmethod
            def my_classmethod(cls):
                return f"patched classmethod for {cls.__name__}"

        PatchClass.apply_patch()

        # Verify that cls is correctly passed as DerivedClass when called
        # through the derived class
        assert (self.TargetClass.my_classmethod() == 
                "patched classmethod for TargetClass")
        assert (self.DerivedClass.my_classmethod() ==
                "patched classmethod for DerivedClass")

    def test_special_attrs_not_patched(self):

        class PatchClass(ArcticPatch[self.TargetClass]):
            my_field = "patched field"
        
        class PatchDerived(ArcticPatch[self.DerivedClass]):
            my_field = "patched field"

        PatchClass.apply_patch()
        PatchDerived.apply_patch()

        assert self.TargetClass._arctic_patches == {"my_field": PatchClass}
        assert self.DerivedClass._arctic_patches == {"my_field": PatchDerived}
