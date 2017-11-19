#!/usr/bin/python

import sys

max_size = 2**22    # one million bodies
current=16384
increment=1024

while current <= max_size:
  for i in xrange(16):
    sys.stdout.write(str(current))
    sys.stdout.write(" ")
    current += increment
  increment *= 2

sys.stdout.flush()
