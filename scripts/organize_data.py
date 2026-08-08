#!/usr/bin/env python
import sys

from sceneledger.cli import main

raise SystemExit(main(["organize", *sys.argv[1:]]))
