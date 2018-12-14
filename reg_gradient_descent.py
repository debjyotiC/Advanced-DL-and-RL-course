import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_samples, w, b = 20, 0.5, 2
xs = np.asarray(range(num_samples))
ys = np.asarray([x * w + b + np.random.normal() for x in range(num_samples)])


class Linear(object):
    def __init__(self):
        self.w = tf.get_variable("w", dtype=tf.float32, shape=[], initializer=tf.zeros_initializer())
        self.b = tf.get_variable("b", dtype=tf.float32, shape=[], initializer=tf.zeros_initializer())

    def __call__(self, x):
        return self.w * x + self.b


xtf = tf.placeholder(tf.float32, [num_samples], "xs")
ytf = tf.placeholder(tf.float32, [num_samples], "ys")
model = Linear()
model_output = model(xtf)

loss = tf.losses.mean_squared_error(ytf,model_output)
grads = tf.gradients(loss, [model.w, model.b])

update_w = tf.assign(model.w, model.w - 0.001*grads[0])
update_b = tf.assign(model.b, model.b - 0.001*grads[1])

update = tf.group(update_w, update_b)


feed_dict = {xtf: xs, ytf: ys}

with tf.train.MonitoredSession() as sess:
    for i in range(500):
        sess.run(update, feed_dict=feed_dict)
        if i in [1, 5, 25, 125, 499]:
            preds = sess.run(model_output, feed_dict=feed_dict)
            plt.plot(xs, preds, label=str(i))
plt.scatter(xs, ys, color='k')
plt.legend()
plt.show()
