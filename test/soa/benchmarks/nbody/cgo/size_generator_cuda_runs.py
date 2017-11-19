#!/usr/bin/python

import sys
import math

size = int(sys.argv[1])
runs = int(2**(22 - math.log(size)/math.log(2))) / 8
if runs < 1:
  runs = 1

sys.stdout.write(str(runs))
sys.stdout.flush()
