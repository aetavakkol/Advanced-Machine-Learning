

import tensorflow as tf

# resolve compatibility issues
tf.compat.v1.disable_eager_execution()


import numpy as np
print(tf.__version__)


tf.compat.v1.reset_default_graph()
a = tf.compat.v1.placeholder(np.float32, (2, 2))
b = tf.compat.v1.Variable(tf.ones((2, 2)))
c = a @ b

s = tf.compat.v1.InteractiveSession()
s.run(tf.compat.v1.global_variables_initializer())
s.run(c, feed_dict={a: np.ones((2, 2))})

tf.compat.v1.reset_default_graph()
x = tf.compat.v1.get_variable("x", shape=(), dtype=tf.float32, trainable=True)
f = x ** 2

optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
step = optimizer.minimize(f, var_list=[x])

tf.compat.v1.trainable_variables()

with tf.compat.v1.Session() as s:  # in this way session will be closed automatically
    s.run(tf.compat.v1.global_variables_initializer())
    for i in range(10):
        _, curr_x, curr_f = s.run([step, x, f])
        print(curr_x, curr_f)

tf.compat.v1.reset_default_graph()
x = tf.compat.v1.get_variable("x", shape=(), dtype=tf.float32)
f = x ** 2
f = tf.compat.v1.Print(f, [x, f], "x, f:")

optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
step = optimizer.minimize(f)

with tf.compat.v1.Session() as s:
    s.run(tf.compat.v1.global_variables_initializer())
    for i in range(10):
        s.run([step, f])


tf.compat.v1.reset_default_graph()
x = tf.compat.v1.get_variable("x", shape=(), dtype=tf.float32)
f = x ** 2


optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
step = optimizer.minimize(f)


tf.compat.v1.summary.scalar('curr_x', x)
tf.compat.v1.summary.scalar('curr_f', f)
summaries = tf.compat.v1.summary.merge_all()


s = tf.compat.v1.InteractiveSession()
summary_writer = tf.compat.v1.summary.FileWriter("logs/1", s.graph)
s.run(tf.compat.v1.global_variables_initializer())
for i in range(10):
    _, curr_summaries = s.run([step, summaries])
    summary_writer.add_summary(curr_summaries, i)
    summary_writer.flush()


