#!/usr/bin/env python
import sys

from sceneledger.cli import main

raise SystemExit(main(["moss-sft", *sys.argv[1:]]))
