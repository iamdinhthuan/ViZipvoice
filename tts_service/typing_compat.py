from __future__ import annotations

import sys
import typing


def ensure_typing_extensions_sentinel() -> None:
    import typing_extensions

    if not hasattr(typing_extensions, "NoExtraItems"):

        class NoExtraItemsType:
            __slots__ = ()

            def __new__(cls):
                return getattr(typing_extensions, "NoExtraItems", None) or object.__new__(cls)

            def __repr__(self) -> str:
                return "typing_extensions.NoExtraItems"

            def __reduce__(self):
                return "NoExtraItems"

        typing_extensions.NoExtraItems = NoExtraItemsType()

    if not hasattr(typing_extensions, "Sentinel"):

        class Sentinel:
            def __init__(self, name: str, repr: str | None = None) -> None:
                self._name = name
                self._repr = repr if repr is not None else f"<{name}>"

            def __repr__(self) -> str:
                return self._repr

            if sys.version_info >= (3, 10):

                def __or__(self, other):
                    return typing.Union[self, other]

                def __ror__(self, other):
                    return typing.Union[other, self]

            def __getstate__(self):
                raise TypeError(f"Cannot pickle {type(self).__name__!r} object")

        typing_extensions.Sentinel = Sentinel
