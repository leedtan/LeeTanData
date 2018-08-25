import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import apply_clipped_optimizer
pd.set_option("display.max_columns", 200)
from sklearn.metrics import r2_score

df = pd.read_csv('ListingsAndSales.csv')
#not sold flag
df['NotSoldFlag'] = 0
df.loc[df['SalesDate'].isnull(), 'NotSoldFlag'] = 1

print('percent not yet sold:', df['NotSoldFlag'].mean())

df.ListingDate = pd.to_datetime(df.ListingDate)

df.SalesDate = pd.to_datetime(df.SalesDate)
df.SalesDate = df.SalesDate.fillna(df.SalesDate.max())

#Get day of dataset for each sample
df['ListingDay'] = (df.ListingDate - df.ListingDate.min()).dt.days
df = df.sort_values('ListingDay')

#calculate days it took to sell the listing if it's sold
df['DaysSold'] = (df.SalesDate - df.ListingDate).dt.days.astype(float) + 1

#loop through the variables and replace missing values with avg and create dummy variables
col_dates = ['ListingDate', 'SalesDate']
for col in df.columns:
    if not col in col_dates:
        if df[col].isnull().sum(axis=0) > 0:
            df[col + "_mv"] = (df[col].isnull())
            col_avg = df.loc[df[col].isnull() == False, col].mean()
            df[col] = df[col].fillna(col_avg)

#Columns to use as regressor
X = df.drop(['DaysSold', 'ListingDate', 'SalesDate', 'NotSoldFlag'], axis=1)

#Column to use as target
Y = df[['DaysSold']].as_matrix().astype(np.float32)

scaler = StandardScaler()
X = pd.DataFrame(
    scaler.fit_transform(X), columns=X.columns).as_matrix().astype(np.float32)
sold = df['NotSoldFlag'].as_matrix().astype(np.float32)

#For numeric stability
EPSILON = 1e-10


class Model():
    def __init__(self, input_size, layer_sizes):
        self.input_size = input_size
        self.layer_sizes = layer_sizes

        self.sold = tf.placeholder(tf.float32, shape=(None))
        self.x = tf.placeholder(tf.float32, shape=(None, input_size))
        self.y = tf.placeholder(tf.float32, shape=(None))

        self.layers = [self.x]
        for layer_size in layer_sizes:
            next_layer = tf.nn.leaky_relu(
                tf.layers.dense(self.layers[-1], layer_size))
            self.layers.append(next_layer)

        self.output = tf.nn.softplus(tf.layers.dense(self.layers[-1], 1))

        self.loss_indicator = (tf.cast(self.output < self.y, tf.float32) *
                               (1 - self.sold) + self.sold)
        loss_numerator = tf.reduce_sum(
            tf.square(self.y - self.output) * self.loss_indicator)
        loss_denominator = (tf.reduce_sum(self.loss_indicator)) + EPSILON
        self.loss = loss_numerator / loss_denominator

        opt_fcn = tf.train.AdamOptimizer()
        self.optimizer = apply_clipped_optimizer(opt_fcn, self.loss)

    def train(self, X, Y, sold, epochs):
        sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
        sess.run(tf.global_variables_initializer())
        n_samples = X.shape[0]
        trn_samples = (n_samples * 4) // 5
        samples = np.arange(n_samples)
        trn_s = np.random.choice(samples, size=trn_samples, replace=False)
        val_samples = [s for s in samples if s not in trn_s]
        X_train, X_test = X[trn_s], X[val_samples]
        Y_train, Y_test = Y[trn_s], Y[val_samples]
        sold_train, sold_test = sold[trn_s], sold[val_samples]
        self.trn_losses = []
        self.val_losses = []
        self.r2_scores = []
        bs = 64
        num_batches = (trn_samples // bs) + 1
        for epoch in range(epochs):
            trn_loss = []
            order = np.arange(trn_samples)
            np.random.shuffle(order)
            for itr in range(trn_samples // bs):
                rows = order[itr * bs:(itr + 1) * bs]
                if itr + 1 == num_batches:
                    rows = order[itr * bs:]
                X_active, Y_active, Sold_active = X_train[rows, :], Y_train[
                    rows], sold_train[rows]
                feed_dict = {
                    self.x: X_active,
                    self.y: Y_active,
                    self.sold: Sold_active
                }
                _, loss, yhat = sess.run(
                    [self.optimizer, self.loss, self.output], feed_dict)
                trn_loss.append(loss)
            if epoch % 2 == 0:
                trn_loss_mean = np.mean(trn_loss)
                self.trn_losses.append(trn_loss_mean)
                feed_dict = {
                    self.x: X_test,
                    self.y: Y_test,
                    self.sold: sold_test
                }
                val_loss, yhat = sess.run([self.loss, self.output], feed_dict)
                self.val_losses.append(val_loss)
                self.r2_scores.append(r2_score(Y_test, yhat))
            if epoch % 10 == 0:
                print('epoch:', epoch, 'train loss: ', trn_loss_mean,
                      'val loss: ', val_loss, 'r2_score:', self.r2_scores[-1])

    def visualize(self, name):
        plt.plot(self.trn_losses, label='train loss')
        plt.plot(self.val_losses, label='test loss')
        plt.title('least square losses')
        plt.legend()
        plt.savefig(name + 'losses.jpg')
        plt.show()
        plt.plot(self.r2_scores, label='validation r2_scores')
        plt.legend()
        plt.title('r2 scores')
        plt.savefig(name + 'r2scores.jpg')
        plt.show()


n_features = X.shape[1]

model = Model(n_features, layer_sizes=[])
model.train(X, Y, sold, epochs=100)
model.visualize('linear_regression')

model = Model(n_features, layer_sizes=[64])
model.train(X, Y, sold, epochs=100)
model.visualize('one_hidden_layer')

model = Model(n_features, layer_sizes=[64, 64])
model.train(X, Y, sold, epochs=100)
model.visualize('two_hidden_layers')
