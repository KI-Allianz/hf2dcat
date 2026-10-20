import sys
import types
from unittest.mock import Mock

import pytest

from huggingface_converter.huggingface_converter.translation_manager import TranslationManager


@pytest.fixture
def argos_backend(monkeypatch):
    """Provide a fake Argos module for one test, without requiring its model."""
    translator = Mock()
    translator.translate.side_effect = lambda text: "DE: " + text
    source = types.SimpleNamespace(code="en", get_translation=Mock(return_value=translator))
    target = types.SimpleNamespace(code="de")
    module = types.ModuleType("argostranslate.translate")
    module.get_installed_languages = Mock(return_value=[source, target])
    package = types.ModuleType("argostranslate")
    package.translate = module
    monkeypatch.setitem(sys.modules, "argostranslate", package)
    monkeypatch.setitem(sys.modules, "argostranslate.translate", module)
    return translator, module


def test_argos_translates_without_google_throttling(argos_backend, monkeypatch):
    def unexpected_throttle(self):
        raise AssertionError("Argos must not use Google throttling")

    monkeypatch.setattr(TranslationManager, "_throttle", unexpected_throttle)
    manager = TranslationManager(provider="argos")
    assert manager.translate_text("Hello") == "DE: Hello"
    assert manager.stats["requests"] == 1
    assert manager.stats["retries"] == 0
    argos_backend[0].translate.assert_called_once_with("Hello")


def test_argos_caches_across_instances(argos_backend, tmp_path):
    cache_path = tmp_path / "translations.json"
    manager = TranslationManager(provider="argos", cache_path=cache_path)
    assert manager.translate_text("Hello") == "DE: Hello"
    assert manager.translate_text("Hello") == "DE: Hello"
    assert manager.stats["cache_hits"] == 1
    manager.save_cache()
    assert cache_path.exists()

    another = TranslationManager(provider="argos", cache_path=cache_path)
    assert another.translate_text("Hello") == "DE: Hello"
    assert argos_backend[0].translate.call_count == 1


def test_argos_never_sends_urls_or_placeholders_to_model(argos_backend):
    manager = TranslationManager(provider="argos")
    original = "Read https://example.org/model for details at https://example.org/docs"
    translated = manager.translate_or_none(original)
    assert translated is not None
    assert translated.count("https://example.org/model") == 1
    assert translated.count("https://example.org/docs") == 1
    assert translated.index("https://example.org/model") < translated.index("https://example.org/docs")
    assert "🔒" not in translated
    for call in argos_backend[0].translate.call_args_list:
        passed = call.args[0]
        assert "http" not in passed
        assert "🔒" not in passed


def test_argos_translates_markdown_link_label_without_touching_url(argos_backend):
    manager = TranslationManager(provider="argos")
    original = "See the [model documentation](https://example.org/docs) for details."
    translated = manager.translate_or_none(original)
    assert translated is not None
    assert "[DE: model documentation](https://example.org/docs)" in translated
    assert translated.count("https://example.org/docs") == 1
    for call in argos_backend[0].translate.call_args_list:
        assert "https://example.org/docs" not in call.args[0]


def test_argos_url_only_returns_no_translation(argos_backend):
    manager = TranslationManager(provider="argos")
    url = "https://example.org/model"
    assert manager.translate_or_none(url) is None
    assert manager.translate_text(url) == url
    argos_backend[0].translate.assert_not_called()


def test_argos_failure_in_one_segment_rejects_whole_translation(argos_backend):
    def translate(text):
        if "details" in text:
            raise RuntimeError("translation failed")
        return "DE: " + text

    argos_backend[0].translate.side_effect = translate
    manager = TranslationManager(provider="argos")
    original = "Read https://example.org/model for details"
    assert manager.translate_or_none(original) is None
    assert manager.translate_text(original) == original
    assert manager.stats["failures"] == 2


def test_argos_failure_translating_markdown_label_rejects_whole_translation(argos_backend):
    def translate(text):
        if text == "model documentation":
            raise RuntimeError("label translation failed")
        return "DE: " + text

    argos_backend[0].translate.side_effect = translate
    manager = TranslationManager(provider="argos")
    original = "See [model documentation](https://example.org/docs)"
    assert manager.translate_or_none(original) is None
    assert manager.translate_text(original) == original


def test_argos_failure_without_urls_falls_back(argos_backend):
    argos_backend[0].translate.side_effect = RuntimeError("translation failed")
    manager = TranslationManager(provider="argos")
    assert manager.translate_or_none("Hello") is None
    assert manager.translate_text("Hello") == "Hello"
    assert manager.stats["failures"] == 2


def test_argos_missing_model_raises(argos_backend):
    argos_backend[1].get_installed_languages.return_value = []
    with pytest.raises(RuntimeError, match="No installed Argos translation model"):
        TranslationManager(provider="argos")


def test_cache_keys_separate_providers():
    google = TranslationManager(enable_translation=False, provider="google")
    argos = TranslationManager(enable_translation=False, provider="argos")
    assert google._cache_key("Hello") != argos._cache_key("Hello")


def test_unknown_provider_rejected():
    with pytest.raises(ValueError, match="provider"):
        TranslationManager(provider="unknown")


@pytest.fixture
def real_argos_manager():
    """Use a real locally installed English-to-German Argos model, if available."""
    argos_translate = pytest.importorskip("argostranslate.translate")
    installed = argos_translate.get_installed_languages()
    source = next((lang for lang in installed if lang.code == "en"), None)
    target = next((lang for lang in installed if lang.code == "de"), None)
    if source is None or target is None or source.get_translation(target) is None:
        pytest.skip("Argos English-to-German model is not installed")
    return TranslationManager(provider="argos")


@pytest.mark.integration
def test_argos_real_english_to_german_translation(real_argos_manager):
    original = "This model is intended for text classification."
    translated = real_argos_manager.translate_or_none(original)
    assert translated is not None
    assert translated.strip()
    assert translated != original


@pytest.mark.integration
def test_argos_real_urls_are_preserved(real_argos_manager):
    original = "Read https://example.org/model for more information."
    translated = real_argos_manager.translate_or_none(original)
    assert translated is not None
    assert translated.count("https://example.org/model") == 1
    assert "🔒" not in translated


@pytest.mark.integration
def test_argos_real_markdown_link_is_preserved(real_argos_manager):
    original = "See [model documentation](https://example.org/docs) for details."
    translated = real_argos_manager.translate_or_none(original)
    assert translated is not None
    assert "https://example.org/docs" in translated
    assert translated.count("https://example.org/docs") == 1
    assert "🔒" not in translated
