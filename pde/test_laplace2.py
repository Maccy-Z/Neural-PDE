from collections import deque

a = deque(maxlen=4)

a.append(1)
a.append(2)
a.append(3)
a.append(4)

print(a[-1])
print(a[-2])
a.append(5)
print(a)

