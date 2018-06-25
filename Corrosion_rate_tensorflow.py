import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score

# Read the datasets
df = pd.read_csv("Datasets_acestic_acid.csv")
# Set x and y and reshape it
x_input = df["Time (min)"].values.reshape(-1,)
y_input = df["Wall – Thickness Loss (µm)"].values.reshape(-1,1)

# Prepare empty list to save an MSE and R^2 from models
MSE = []
R2_score = []
# Empty list for calculated new value of linear regression
Y_test =[]

# set a loop for 3 different models
for x in range(3):
    n = x+1
    # Set a variables for weights
    W = tf.Variable(tf.random_normal([n,1]), name='weight')
    # Set variables for bias
    b = tf.Variable(tf.random_normal([1]), name='bias')

    # Set a placeholder for input data
    X=tf.placeholder(tf.float32,shape=[None,n])
    Y=tf.placeholder(tf.float32,shape=[None, 1])


    #preparing the data
    def modify_input(x,x_size,n_value):
        x_new=np.zeros([x_size,n_value])
        for i in range(n):
            x_new[:,i]=np.power(x,(i+1))
            x_new[:,i]=x_new[:,i]/np.max(x_new[:,i])
        return x_new

    x_modified=modify_input(x_input,x_input.size,n)
    # Create model
    Y_pred=tf.add(tf.matmul(X,W),b)

    #algortihm
    # Calculate the loss MSE
    loss = tf.losses.mean_squared_error(Y, Y_pred)
    #training algorithm
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    #initializing the variables
    init = tf.global_variables_initializer()

    #starting the session
    sess = tf.Session()
    sess.run(init)
    # Reset the graph
    tf.reset_default_graph()
    # stes number of epochs
    epoch= 100000
    # Run the model
    for step in range(epoch):
         _, c=sess.run([optimizer, loss], feed_dict={X: x_modified, Y: y_input})
         if step%1000==0 :
            print(c)
    # print model paramters
    print("Model paramters:" )
    print(sess.run(W))
    print("Bias:%f" %sess.run(b))
    # Calculate the prediction value (linear regression points)
    y_test=sess.run(Y_pred, feed_dict={X:x_modified})
    # Calculate the R^2
    r2 = r2_score(y_input, y_test)
    print(r2)
    # Append data to empty list
    Y_test.append(y_test)
    MSE.append(c)
    R2_score.append(r2)
# Print the MSE and R^2
print(MSE)
print(R2_score)
# Change list to array
y_test = np.array(Y_test)

# plot results
# set a plot size
plt.figure(figsize=(16,8))
# plot datasets points
plt.scatter(x_input, y_input, label='training points', color='orange',s=5)
# plot simple linear regression
plt.plot(x_input, y_test[0],
         label='linear (n=1), $R^2=%.5f$ $MSE=%.5f$' % (R2_score[0], MSE[0]),
         color='black',
         lw=2,
         linestyle=':')
# plot quadratic linear regression
plt.plot(x_input, y_test[1],
         label='quadratic (n=2), $R^2=%.5f$  $MSE=%.5f$' % (R2_score[1], MSE[1]),
         color='red',
         lw=2,
         linestyle='-')
# plot cubic linear egression
plt.plot(x_input, y_test[2],
         label='cubic (n=3), $R^2=%.5f$  $MSE=%.5f$' % (R2_score[2], MSE[2]),
         color='green',
         lw=2,
         linestyle='--')
# add title to plot
plt.title('Tensorflow model')
# set x and y labels
plt.xlabel('TIME [min]')
plt.ylabel('WALL THICKNESS [µm]')
# Add legend
plt.legend(loc='upper right')
# save plot
plt.savefig('tensorflow.png',bbox_inches='tight',dpi=600)
plt.show()
