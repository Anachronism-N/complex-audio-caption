#!/usr/bin/env python
import sys

from sceneledger.cli import main

raise SystemExit(main(["carc", *sys.argv[1:]]))
