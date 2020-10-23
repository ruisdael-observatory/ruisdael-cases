import numpy as np

itot = 2
jtot = 3
ktot = 4

# Create dummy data
a = np.arange(itot*jtot*ktot).reshape((itot,jtot,ktot))
b = np.arange(itot*jtot).reshape((itot,jtot))

for i in range(itot):
    for j in range(jtot):
        for k in range(ktot):
            print(i,j,k,a[i,j,k])

for i in range(itot):
    for j in range(jtot):
        print(i,j,b[i,j])

# Write to binary file
a.T.tofile('bla1.bin')
b.T.tofile('bla2.bin')

