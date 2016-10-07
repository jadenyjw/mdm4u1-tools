import math
import numpy
import matplotlib.pyplot as plt

f = open('data.txt', 'r')
lines = f.readlines()

sum_x = 0.0
sum_y = 0.0
sum_xy = 0.0
sum_x_squared = 0.0
sum_y_squared = 0.0

x_data = [i.split(' ', 1)[0] for i in lines]
y_data = [i.split(' ', 1)[1].rstrip() for i in lines]

for i in lines:
    x = float(i.split(' ', 1)[0])
    y = float(i.split(' ', 1)[1].rstrip())

    sum_x += x
    sum_y += y
    sum_xy += x * y
    sum_x_squared += math.pow(x, 2)
    sum_y_squared += math.pow(y, 2)

n = len(lines)

r = ((n * sum_xy) - (sum_x * sum_y)) / math.sqrt((n * sum_x_squared - math.pow(sum_x, 2)) * (n * sum_y_squared - math.pow(sum_y, 2)))
a = ((n * sum_xy) - (sum_x * sum_y)) / (n * sum_x_squared - math.pow(sum_x, 2))
b = (sum_y / n) - (a * (sum_x / n))

train_X = numpy.asarray(x_data).astype('float32')
train_Y = numpy.asarray(y_data).astype('float32')

print("r=" + str(r), "a=" + str(a), "b=" + str(b))

plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.plot(train_X, a * train_X + b, label='Fitted line')
plt.legend()
plt.show()
