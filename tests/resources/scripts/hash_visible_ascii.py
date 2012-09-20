#/usr/bin/python
import md5

f_pre = open('md5_visible_3_pre', 'w')
f_hash = open('md5_visible_3_hash', 'w')

for i in range(32,127):
    for j in range(32,127):
        for k in range(32,127):
            
	    f_pre.write(chr(i)+chr(j)+chr(k)+'\n')
            f_hash.write(md5.new(chr(i)+chr(j)+chr(k)).hexdigest()+'\n')

f_pre.close()
f_hash.close()
