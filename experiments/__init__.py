import sys
import os

modpath = os.path.realpath(os.path.abspath(".."))
print(f"Adding '{modpath}' to path...")
sys.path.append(modpath)
