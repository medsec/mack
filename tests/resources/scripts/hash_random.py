#!/usr/bin/python
import hashlib
import random

f_pre = open('md5_visible_rand_3_pre', 'w')
f_hash = open('md5_visible_rand_3_hash', 'w')

for x in range(100000):
	i = random.randint(32, 126)
	j = random.randint(32, 126)
	k = random.randint(32, 126)      
	f_pre.write(chr(i)+chr(j)+chr(k)+'\n')
        f_hash.write(hashlib.md5(chr(i)+chr(j)+chr(k)).hexdigest()+'\n')

f_pre.close()
f_hash.close()
