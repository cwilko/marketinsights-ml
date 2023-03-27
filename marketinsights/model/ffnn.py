import numpy as np
import tensorflow as tf
import quantutils.dataset.ml as mlutils
import quantutils.dataset.pipeline as ppl
from marketinsights.api.model import MarketInsightsModel


class Perceptron(MarketInsightsModel):

    def __init__(self, name, featureCount, labelCount, modelConfig):
        super().__init__(name)

        self.OPT_CNF = modelConfig['optimizer']
        self.NTWK_CNF = modelConfig['network']
        self.TRAIN_CNF = modelConfig['training']

        HIDDEN_UNITS = self.NTWK_CNF["hidden_units"]
        BIAS = self.NTWK_CNF["bias"]
        STDEV = self.NTWK_CNF["weights"]["stdev"]
        SEED = self.NTWK_CNF["weights"]["seed"]

        self.NUM_FEATURES = featureCount
        self.NUM_LABELS = labelCount

        self.losses = []

        # Initialize model parameters
        self.Theta1 = tf.Variable(tf.random.normal([HIDDEN_UNITS, self.NUM_FEATURES], stddev=STDEV, seed=SEED, dtype=tf.float32))
        self.Theta2 = tf.Variable(tf.random.normal([self.NUM_LABELS, HIDDEN_UNITS], stddev=STDEV, seed=SEED, dtype=tf.float32))
        self.bias = tf.Variable(tf.constant(BIAS, shape=[self.NUM_LABELS], dtype=tf.float32))
        self.lam = tf.constant(0.001, tf.float32)

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32)])
    def __call__(self, x):
        layer1 = tf.nn.sigmoid(tf.matmul(x, tf.transpose(self.Theta1)))
        output = tf.nn.bias_add(tf.matmul(layer1, tf.transpose(self.Theta2)), self.bias)
        return output

    def loss(self, y_pred, y):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_pred))

        # Regularization using L2 Loss function
        regularizer = tf.nn.l2_loss(self.Theta1) + tf.nn.l2_loss(self.Theta2)
        reg = (self.lam / tf.cast(tf.shape(y)[0], tf.float32)) * regularizer
        loss_reg = loss + reg

        return loss_reg

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32)])
    def predictions(self, x):
        return tf.nn.sigmoid(self(x))

    def train(self, train_x, train_y, learning_rate=0.01, epochs=-1, threshold=-1):

        epochs = epochs if epochs >= 0 else self.TRAIN_CNF["iterations"]
        threshold = threshold if threshold >= 0 else self.TRAIN_CNF["threshold"]

        train_x = tf.constant(train_x)
        train_y = tf.constant(train_y)
        learning_rate = tf.constant(learning_rate, tf.float32)

        # Format training loop
        for epoch in range(epochs):

            with tf.GradientTape() as tape:
                batch_loss = self.loss(self(train_x), train_y)

            # Update parameters with respect to the gradient calculations
            grads = tape.gradient(batch_loss, self.variables)
            for g, v in zip(grads, self.variables):
                v.assign_sub(learning_rate * g)

            # Keep track of model loss per epoch
            loss = self.loss(self(train_x), train_y)
            self.losses.append(loss)
            if epoch % 100 == 0:
                print(f'Mean squared error for step {epoch}: {loss.numpy():0.3f}')
                print(f'Precision {mlutils.evaluate(ppl.onehot(self.predictions(train_x).numpy()), ppl.onehot(train_y.numpy()), display=False):0.2f}')

    def getSignatures(self):
        call_sig = self.__call__.get_concrete_function(tf.TensorSpec([None, None], tf.float32))
        predict_sig = self.predictions.get_concrete_function(tf.TensorSpec([None, None], tf.float32))
        return {"serving_default": call_sig, "predictions": predict_sig}
