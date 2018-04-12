import sys
data = sys.stdin.readline().strip()
lst1 = sys.stdin.readline().strip()
lst2 = sys.stdin.readline().strip()
data1 = data.split()
n = int(data1[0])
m = int(data1[1])
lst1 =lst1.split()
lst11 = []
for i in lst1:
    lst11.append(int(i))
print(lst11)
lst2 = lst2.split()
lst22 = []
for i in lst2:
    lst22.append(int(i))
ret = lst22 + lst11
ret.sort()
ret1=[]
for i in ret:
    ret1.append(str(i))

print(ret1)

print(" ".join(ret1))