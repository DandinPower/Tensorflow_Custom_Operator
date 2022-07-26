import matplotlib.pyplot as plt

distribution = []
with open("distribution.txt", "r") as txt_file:
    readline=txt_file.read().splitlines()
    for i in range(len(readline)):
        distribution.append(int(readline[i]))

plt.plot(distribution)
plt.xlabel('Flip bits')
plt.ylabel('Nums')
plt.title('Random Flip Distribution')
plt.show()
