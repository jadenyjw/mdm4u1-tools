import math

f = open('data.txt', 'r')
lines = f.readlines()

sum_x = 0
sum_y = 0
sum_xy = 0
sum_x_squared = 0
sum_y_squared = 0

for i in lines:
    x = int(i.split(' ', 1)[0])
    y = int(i.split(' ', 1)[1].rstrip())

    sum_x += x
    sum_y += y
    sum_xy += x * y
    sum_x_squared += math.pow(x, 2)
    sum_y_squared += math.pow(y, 2)

n = len(lines)

r = ((n * sum_xy) - (sum_x * sum_y)) / math.sqrt((n * sum_x_squared - math.pow(sum_x, 2)) * (n * sum_y_squared - math.pow(sum_y, 2)))
a = ((n * sum_xy) - (sum_x * sum_y)) / (n * sum_x_squared - math.pow(sum_x, 2))
b = (sum_y / n) - (a * (sum_x / n))
print("r=" + str(r), "a=" + str(a), "b=" + str(b))
