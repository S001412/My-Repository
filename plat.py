import platform
a = 0
for i in range(9999):
        a += i
        print("Finish job, result=%i" % a)
        print("This is", platform.system())

