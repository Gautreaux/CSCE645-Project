from typing import Generator
from env import DUMP_DIRECTORY, DUMP_LOCATION_SLDPRT
import time
import os
from shutil import copyfile
from math import floor

# return true if this is a path to a temp file
def isTempPath(p):
    return "$" in p

# walk this root dir and all sub dirs
# generate all the files on the drive with a specific extension
def generateFilesByExtension(extension:str, root_dir:str="/", report=True) -> Generator[str, None, None]:
    start_time = time.time()
    total_files = 0
    matched_files = 0

    for root, dirs, files in os.walk("/", followlinks=False):
        for f in files:
            total_files += 1
            if f.rpartition(".")[-1] == extension:
                matched_files += 1
                p = "/".join((root, f))

                if not isTempPath(p):
                    yield p

        if report:
            elapsed = time.time() - start_time
            if elapsed > .5:
                print(f"Completed directory {root}, total files: {total_files}, matched files: {matched_files}")
                start_time = time.time()


# group all the sldprt files found into the dump folder
def doCoalesceSldprt(rootdir: str = "/"):
    ctr = 0
    log_name = f"coalesceSldprt.{floor(time.time())}.log"
    with open(f"{DUMP_DIRECTORY}{log_name}", 'w') as outfile:
        for f in generateFilesByExtension("sldprt"):
            ctr += 1
            print(f, file=outfile)
    print(f"Found {ctr} valid file paths")



doCoalesceSldprt()