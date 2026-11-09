"""Entry point: python -m rc_sdk [--manifest manifest.json] [--policy module:Class]"""

from .server import main

if __name__ == "__main__":
    main()
