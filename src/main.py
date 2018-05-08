import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#data = input_data.read_data_sets("MNIST_data/",one_hot=True)
def _get_data(filename,label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded,[28,28])
    return image_resized,label

def fmt(i):
    if i > 99:
        return str(i)
    elif i > 9:
        return '0' + str(i)
    else:
        return '00' + str(i)

filenames = ['IMG_0' + fmt(i) + '.JPG' for i in range(2,262)] 
labels = tf.constant([0 for i in range(200)] + [1 for i in range(60)])
data = tf.data.Dataset.from_tensor_slices((filenames,labels))
data = data.map(_get_data)

training_epochs = 15
batch_size = 100
display_step = 1

n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10

x = tf.placeholder('float',[None,n_input])
y = tf.placeholder('float',[None,n_classes])

def next_batch(num,data,labels):
    # Adapted from https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data
    idx = np.arange(0,n_input)
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle),np.asarray(labels_shuffle)

def multilayer_perceptron(x,weights,biases):
    layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    return tf.matmul(layer_2,weights['out']) + biases['out']

weights = {
            'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
            'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
            'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes])),
        }
biases = {
            'b1':tf.Variable(tf.random_normal([n_hidden_1])),
            'b2':tf.Variable(tf.random_normal([n_hidden_2])),
            'out':tf.Variable(tf.random_normal([n_classes])),
        }

pred = multilayer_perceptron(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(260/batch_size)
        for i in range(total_batch):
            batch_x,batch_y = next_batch(batch_size,x,y)
            _,c = sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
            avg_cost += c / total_batch
        if epoch % display_step == 0:
            print("Epoch:","%04d" % (epoch+1),"cost=","{:.9f}".format(avg_cost))
    print("Optimization Finished")
    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
    print('Accuracy:',accuracy.eval({x:data.test.images,y:data.test.labels}))
