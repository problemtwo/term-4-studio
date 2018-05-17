import cv2
import random

def fmt(i):
    if i < 10:
        return '000' + str(i)
    elif i < 100:
        return '00' + str(i)
    return '0' + str(i)

def read_images():
    images = []
    for i in range(2,262):
        print('Reading image ' + str(i-1) + ' of 260')
        images.append(cv2.imread('../training/images/IMG_' + fmt(i) + '.JPG'))
    return images

def read_labels():
    f = open('../training/labels-1.txt','r')
    return list(f)
        

def matadd(x,y):
    res = []
    for j in min(len(x),len(y)):
        row = []
        for i in min(len(x[j]),len(y[j])):
            row.append(x[j][i] + y[j][i])
        res.append(row)
    return res

def matsub(x,y):
    res = []
    for j in min(len(x),len(y)):
        row = []
        for i in min(len(x[j]),len(y[j])):
            row.append(x[j][i] - y[j][i])
        res.append(row)
    return res

def reg_matmul(x,y):
    # Used for matrices of dimensions 2x2 and under.
    z = [[0 for i in range(len(x))] for j in range(len(y[0]))]
    for i in range(len(x)):
        for j in range(len(y[0])):
            for k in range(len(y)):
                z[i][j] += x[i][k] * y[k][j]
    return z

def strassens_matmul(x,y):
   """
    [a,b  * [e,f  =
     c,d]    g,h]

    [p5+p4-p2+p6,p1+p2
     p3+p4,p1+p5-p3-p7]

    where:

    p1 = a(f - h)
    p2 = (a+b)h
    p3 = (c+d)e
    p4 = d(g-e)
    p5 = (a+d)(e+h)
    p6 = (b-d)(g+h)
    p7 = (a-c)(e+f)

   """
   if(len(x) <= 2 and len(x[0]) <= 2 and len(y) <= 2 and len(y[0]) <= 2):
       return reg_matmul(x,y)
   half_x_w = math.floor(len(x[0]) / 2)
   half_x_h = math.floor(len(x) / 2)
   half_y_w = math.floor(len(y[0]) / 2)
   half_y_h = math.floor(len(y) / 2)
   xs = []
   xs.append([x[i][:half_x_w] for i in range(half_x_h)])
   xs.append([x[i][half_x_w:] for i in range(half_x_h)])
   xs.append([x[i][:half_x_w] for i in range(half_x_h,len(x))])
   xs.append([x[i][half_x_w:] for i in range(half_x_h,len(x))])
   ys = []
   ys.append([y[i][:half_y_w] for i in range(half_y_h)])
   ys.append([y[i][half_y_w:] for i in range(half_y_h)])
   ys.append([y[i][:half_y_w] for i in range(half_y_h,len(y))])
   ys.append([y[i][half_y_w:] for i in range(half_y_h,len(y))])
   p1 = strassens_matmul(xs[0],matsub(ys[1],ys[3]))
   p2 = strassens_matmul(ys[3],matadd(xs[0],xs[1]))
   p3 = strassens_matmul(ys[0],matadd(xs[2],xs[3]))
   p4 = strassens_matmul(xs[3],matsub(ys[2],ys[0]))
   p5 = strassens_matmul(matadd(xs[0],xs[3]),matadd(ys[0],ys[3]))
   p6 = strassens_matmul(matsub(xs[1],xs[3]),matadd(ys[2],ys[3]))
   p7 = strassens_matmul(matsub(xs[0],xs[2]),matadd(ys[0],ys[1]))
   q1 = matadd(matsub(matadd(p5,p4),p2),p6)
   q2 = matadd(p1,p2)
   q3 = matadd(p3,p4)
   q4 = matsub(matsub(matadd(p1,p5),p3),p7)
   res = []
   for i in range(min(len(q1),len(q2))):res.append(q1[i] + q2[i])
   for j in range(min(len(q3),len(q4))):res.append(q3[i] + q4[i])
   return res

# relu and dRelu borrowed from https://gist.github.com/yusugomori/cf7bce19b8e16d57488a
def relu(a):
    return a * (a > 0)

def dRelu(a):
    return 1. * (a > 0)

def multilayer_perceptron(x,weights,biases):
    print('Starting network...')
    a = matadd(strassens_matmul(x,weights[0]),biases[0])
    a = relu(a)
    print('Done with hidden layer 1')
    b = matadd(strassens_matmul(a,weights[1]),biases[1])
    b = relu(b)
    print('Done with hidden layer 2')
    return matadd(strassens_matmul(b,weights[2]),biases[2])

images = read_images()
labels = read_labels()

image_size = 2448 * 3264
hidden_1_size = 256
hidden_2_size = 256
classes_size = 2

weights = [
            [[1 for i in range(image_size)] for j in range(hidden_1_size)],
            [[1 for i in range(hidden_1_size)] for j in range(hidden_2_size)],
            [[1 for i in range(hidden_2_size)] for j in range(classes_size)],
        ]
biases = [
            [1 for i in range(hidden_1_size)],
            [1 for i in range(hidden_2_size)],
            [1 for i in range(classes_size)],
        ]

prediction = multilayer_perceptron(images,weights,biases)
print(prediction)
