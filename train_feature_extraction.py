import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle

# Load traffic signs data.
num_classes = 43
with open('./train.p', 'rb') as f:
    data = pickle.load(f)

features = data['features']
labels = data['labels']

# Split data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.33, random_state=0)

# Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
x_resized = tf.image.resize_images(x, (227, 227))

y = tf.placeholder(tf.int64, (None))
#one_hot_y = tf.one_hot(y, 43)

# pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(x_resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], num_classes)
whts = tf.Variable(tf.truncated_normal(shape, mean=0, stddev=0.1))
biases = tf.Variable(tf.zeros(num_classes))
logits = tf.nn.xw_plus_b(fc7, whts, biases)

# Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y)

# Loss function cross entropy reduction and L2 Regularization on weights
loss_operation = tf.reduce_mean(cross_entropy)
regularizers = tf.nn.l2_loss(whts)
loss = tf.reduce_mean(loss_operation + 0.01 * regularizers)

optimizer = tf.train.AdamOptimizer(learning_rate = 0.0005)
training_operation = optimizer.minimize(loss, var_list=[whts,biases])

# model evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), y)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
BATCH_SIZE = 128
EPOCHS = 10

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

#  Train and evaluate the feature extraction model.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        
        training_accuracy = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

