import logging
from types import MethodType
from typing import Type, TypeVar, Generic

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ArcticPatch(Generic[T]):

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Ensure that subclasses are created using the subscript syntax.
        if not hasattr(cls, '_arctic_patch_target'):
            raise TypeError("Sub-classes of ArcticPatch must be defined as "
                            "ArcticPatch[Target] to specify a patch target.")

    @classmethod
    def __class_getitem__(cls, target: Type[T]) -> Type:
        # The dynamic type created here will carry the target class as
        # _arctic_patch_target.
        return type(f"{cls.__name__}[{target.__name__}]", (cls,),
                    {'_arctic_patch_target': target})

    @classmethod
    def apply_patch(cls):
        """
        Patches the target class (stored in _arctic_patch_target) by replacing its attributes with those
        defined on the current class (self). It iterates over the class __dict__ and copies all attributes
        except for special names and the '_arctic_patch_target' itself.
        """
        target = cls._arctic_patch_target

        if "_arctic_patches" not in target.__dict__:
            target._arctic_patches = {}

        for name, attr in cls.__dict__.items():

            # Skip special names and the '_arctic_patch_target' itself
            if name in ("_arctic_patch_target", "__dict__", "__weakref__",
                        "__module__", "__doc__", "__parameters__",):
                continue

            # Check if the attribute has already been patched
            if name in target._arctic_patches:
                patch = target._arctic_patches[name]
                raise ValueError(f"{target.__name__}.{name} is already "
                                 f"patched by {patch.__name__}!")
            target._arctic_patches[name] = cls

            # If classmethod, re-bind it to the target
            if isinstance(attr, MethodType):
                attr = MethodType(attr.__func__, target)

            # Patch the target with the new attribute
            replace = hasattr(target, name)
            setattr(target, name, attr)
            action = "replaced by" if replace else "added by"
            logger.info(f"{target.__name__}.{name} {action} {cls.__name__}")
