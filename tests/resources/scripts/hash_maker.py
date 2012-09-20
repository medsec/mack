#!/usr/bin/python

import hashlib

for i in range(32,127):
    for j in range(32,127):
        for k in range(32,127):
            # print chr(i)+chr(j)+chr(k)
            print hashlib.sha256(chr(i)+chr(j)+chr(k)).hexdigest()
