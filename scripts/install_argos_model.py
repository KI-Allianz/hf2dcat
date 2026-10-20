"""Install an Argos Translate language model (default: English -> German).

Usage:
    python install_argos_model.py
    python install_argos_model.py --from en --to fr
"""

import argparse
import logging
import sys

import argostranslate.package
import argostranslate.translate


def get_installed_translation(from_code: str, to_code: str):
    """Return the installed from_code -> to_code translation, or None."""
    languages = argostranslate.translate.get_installed_languages()
    src = next((lang for lang in languages if lang.code == from_code), None)
    dst = next((lang for lang in languages if lang.code == to_code), None)
    return src.get_translation(dst) if src and dst else None


def is_installed(from_code: str, to_code: str) -> bool:
    """Pre-check: True if the model is installed. A model that cannot be read
    counts as missing, so the script goes on and (re)installs it."""
    try:
        return get_installed_translation(from_code, to_code) is not None
    except Exception as exc:
        print(
            f"Warning: installed Argos {from_code}->{to_code} model could not be read "
            f"({type(exc).__name__}: {exc}); reinstalling.",
            file=sys.stderr,
        )
        return False


def verify(from_code: str, to_code: str) -> None:
    """Post-check: load the model and translate a test sentence. Errors are raised,
    not swallowed, so a model that is still broken after installing is reported."""
    translation = get_installed_translation(from_code, to_code)
    if translation is None:
        raise RuntimeError(f"Argos {from_code}->{to_code} model not found after installing.")
    # get_translation() only looks the model up; translate() actually loads it.
    if not translation.translate("Hello world.").strip():
        raise RuntimeError(f"Argos {from_code}->{to_code} model returned an empty test translation.")


def install(from_code: str, to_code: str) -> None:
    if is_installed(from_code, to_code):
        print(f"Argos {from_code}->{to_code} model is already installed.")
        return

    print("Updating the Argos package index ...")
    argostranslate.package.update_package_index()

    package = next(
        (
            p for p in argostranslate.package.get_available_packages()
            if p.from_code == from_code and p.to_code == to_code
        ),
        None,
    )
    if package is None:
        raise RuntimeError(f"No Argos {from_code}->{to_code} model in the package index.")

    print(f"Downloading {package} ...")
    argostranslate.package.install_from_path(package.download())

    # Check that the model really is usable, not only that the install call returned.
    verify(from_code, to_code)
    print(f"Argos {from_code}->{to_code} model installed successfully.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Install an Argos Translate language model.")
    parser.add_argument("--from", dest="from_code", default="en", help="source language code (default: en)")
    parser.add_argument("--to", dest="to_code", default="de", help="target language code (default: de)")
    args = parser.parse_args()

    # argostranslate logs its internals at INFO; keep the script output readable.
    logging.getLogger("argostranslate").setLevel(logging.WARNING)
    logging.getLogger("argostranslate.utils").setLevel(logging.WARNING)

    try:
        install(args.from_code, args.to_code)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
