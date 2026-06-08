import sys
import types
import typing


def test_ensure_typing_extensions_sentinel_installs_compatible_polyfill(monkeypatch):
    fake_typing_extensions = types.ModuleType("typing_extensions")
    monkeypatch.setitem(sys.modules, "typing_extensions", fake_typing_extensions)

    from tts_service.typing_compat import ensure_typing_extensions_sentinel

    ensure_typing_extensions_sentinel()

    missing = fake_typing_extensions.Sentinel("MISSING")
    assert repr(missing) == "<MISSING>"
    assert missing | str == typing.Union[missing, str]
    assert int | missing == typing.Union[int, missing]


def test_ensure_typing_extensions_sentinel_installs_no_extra_items(monkeypatch):
    fake_typing_extensions = types.ModuleType("typing_extensions")
    monkeypatch.setitem(sys.modules, "typing_extensions", fake_typing_extensions)

    from tts_service.typing_compat import ensure_typing_extensions_sentinel

    ensure_typing_extensions_sentinel()

    assert repr(fake_typing_extensions.NoExtraItems) == "typing_extensions.NoExtraItems"
    assert fake_typing_extensions.NoExtraItems.__reduce__() == "NoExtraItems"
