#!/usr/bin/env python
import sys

from sceneledger.cli import main

raise SystemExit(main(["download", *sys.argv[1:]]))
