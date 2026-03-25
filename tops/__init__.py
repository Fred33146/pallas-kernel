try:
    from tops._version import __version__
except ImportError:
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version("tops")
    except PackageNotFoundError:
        __version__ = "unknown"

__all__ = ["__version__"]
