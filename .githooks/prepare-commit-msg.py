#!/usr/bin/env python

import sys

commit_msg_filepath = sys.argv[1]

print("Start script", commit_msg_filepath)

with open(commit_msg_filepath, "w", newline="\n") as handle:
    handle.write("Start script")
