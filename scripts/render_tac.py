#!/usr/bin/env python
import sys

from sceneledger.cli import main

raise SystemExit(main(["render", *sys.argv[1:]]))
