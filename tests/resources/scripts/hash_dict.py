#/usr/bin/python
import md5
import random

#first dict word
f_pre = open('targets/dict_first_pre', 'w')
f_hash = open('targets/dict_first_hash', 'w')
dict = open('dict/en.dict', 'r')


l = dict.readlines()
dict.close()

f_pre.write(l[0].strip())
f_hash.write(md5.new(l[0].strip()).hexdigest())
f_pre.close()
f_hash.close()

#last dict word
f_pre = open('targets/dict_last_pre', 'w')
f_hash = open('targets/dict_last_hash', 'w')
f_pre.write(l[len(l)-1].strip())
f_hash.write(md5.new(l[len(l)-1].strip()).hexdigest())
f_pre.close()
f_hash.close()


#random dict word (not last,first)
r= random.randint(1, len(l)-2)
f_pre = open('targets/dict_rand_pre', 'w')
f_hash = open('targets/dict_rand_hash', 'w')
f_pre.write(l[r].strip())
f_hash.write(md5.new(l[r].strip()).hexdigest())
f_pre.close()
f_hash.close()
