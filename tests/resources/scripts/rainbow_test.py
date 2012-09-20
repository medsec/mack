#/usr/bin/python

def hash(s):
    tmp = 0;
    for i in s:
        tmp ^= ord(i)
    return tmp

# 3*row + ((wordlength-1) * 255)
def reduce(s, row):
    return s

print hash("ab")
