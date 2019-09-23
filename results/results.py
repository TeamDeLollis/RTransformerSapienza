import matplotlib.pyplot as plt

file2 = 'ner2'
loss2 = []
accuracy2 = []
f12 = []

with open(file2) as f:
    for line in f:
        split = line.split()
        loss2.append(float(split[0]))
        accuracy2.append(float(split[1]))
        f12.append(float(split[2]))

plt.plot(loss2, 'o-')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

plt.plot(accuracy2, 'o-')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

plt.plot(f12, 'o-')
plt.ylabel('F1')
plt.xlabel('Epoch')
plt.show()


file3 = 'ner3'
loss3 = []
accuracy3 = []
f13 = []

with open(file3) as f:
    for line in f:
        split = line.split()
        loss3.append(float(split[0]))
        accuracy3.append(float(split[1]))
        f13.append(float(split[2]))

plt.plot(loss3, 'o-')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

plt.plot(accuracy3, 'o-')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

plt.plot(f13, 'o-')
plt.ylabel('F1')
plt.xlabel('Epoch')
plt.show()
