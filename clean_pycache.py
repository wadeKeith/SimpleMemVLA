import os
import shutil

root_dir = "./"

for dirpath, dirnames, filenames in os.walk(root_dir):
    if "__pycache__" in dirnames:
        pycache_dir = os.path.join(dirpath, "__pycache__")

        shutil.rmtree(pycache_dir)

        print(f"Removed: {pycache_dir}")



