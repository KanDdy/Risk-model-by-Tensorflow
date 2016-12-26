import pandas as pd
import numpy as np
import tensorflow as tf

from itertools import cycle

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from scipy import interp

num_classes = 2


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

FEATURE_COLUMNS = ["YQ", "AGE", "SEX", "KHPJ", "GXPJ", "CKYEPJ",
                   "ZL", "ZHS", "ZCPS"]

data = np.array(pd.read_csv('data_logistic.csv'))
labels = data[:, 1].astype(np.int32)
labels[labels > 0] = 1


counts = [0, 0, 0, 0, 0, 0, 0]
for i in range(len(labels)):
    if labels[i] == 0:
      counts[0] += 1
    elif labels[i] >= 1:
      if labels[i] == 1: counts[1] += 1
      elif labels[i] == 2: counts[2] += 1
      elif labels[i] == 3: counts[3] += 1
      elif labels[i] == 4: counts[4] += 1
      elif labels[i] == 5: counts[5] += 1
      else:
          counts[6] += 1


features = data[:, 2:33].astype(np.int32)
features_1 = data[:, 2:7]
features_2 = data[:, 24:27]
features_3 = np.concatenate((features_1, features_2), axis=1)

labels_none_YQ = labels[0:counts[0]]
labels_alr_YQ = labels[counts[0]:]
features_none_YQ = features_3[0:counts[0], :]
features_alr_YQ = features_3[counts[0]:, :]

concentration = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

ran = np.random.choice(np.arange(counts[0]), int((1-concentration[5])/concentration[5] * len(labels_alr_YQ)))

labels_ = np.concatenate((labels_none_YQ[ran], labels_alr_YQ))
features_ = np.concatenate((features_none_YQ[ran], features_alr_YQ))

num_samples, num_features = features_.shape
#features_ = np.c_[features_, random_state.randn(num_samples, 200 * num_features)]

# Split the datasets into train and test part respectively
X_train, X_test, y_train, y_test = train_test_split(features_, labels_, test_size=0.3, random_state=0)

TRAINING = np.concatenate((y_train.reshape(len(y_train), 1), X_train), axis=1)
TEST = np.concatenate((y_test.reshape(len(y_test), 1), X_test), axis=1)
pd.DataFrame(TRAINING).to_csv('training2.csv', index=False)
pd.DataFrame(TEST).to_csv('test2.csv', index=False)


# Load datasets
df_train = pd.read_csv('training2.csv', names=FEATURE_COLUMNS, header=None, skiprows=1).astype(np.int)
df_test = pd.read_csv('test2.csv', names=FEATURE_COLUMNS, header=None, skiprows=1).astype(np.int)

LABEL_COLUMN = "label"
df_train[LABEL_COLUMN] = df_train["YQ"].astype(np.int)
df_test[LABEL_COLUMN] = df_test["YQ"].astype(np.int)

CATEGORICAL_COLUMNS = ["AGE", "SEX",
                       "KHPJ", "GXPJ", "CKYEPJ"]
CONTINUOUS_COLUMNS = ["ZL", "ZHS", "ZCPS"]

df_train[CATEGORICAL_COLUMNS] = df_train[CATEGORICAL_COLUMNS].astype(np.str)
df_test[CATEGORICAL_COLUMNS] = df_test[CATEGORICAL_COLUMNS].astype(np.str)


# Converting Data into Tensors
def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols.items() + categorical_cols.items())
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label


def train_input_fn():
  return input_fn(df_train)


def eval_input_fn():
  return input_fn(df_test)

# Categorical columns
AGE = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="AGE", hash_bucket_size=10)
SEX = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="SEX", hash_bucket_size=2)
KHPJ = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="KHPJ", hash_bucket_size=5)
GXPJ = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="GXPJ", hash_bucket_size=11)
CKYEPJ = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="CKYEPJ", hash_bucket_size=11)

# Continuous features
ZL = tf.contrib.layers.real_valued_column("ZL")
ZHS = tf.contrib.layers.real_valued_column("ZHS")
ZCPS = tf.contrib.layers.real_valued_column("ZCPS")


model_dir = 'risk_model_linear/Binary'

feature_columns = [AGE, SEX, GXPJ, CKYEPJ, KHPJ, ZL, ZHS, ZCPS]

model = tf.contrib.learn.LinearClassifier(feature_columns=feature_columns,
                                          optimizer=tf.train.FtrlOptimizer(
                                          learning_rate=0.01,
                                          l1_regularization_strength=1.0,
                                          l2_regularization_strength=1.0
                                      ), model_dir=model_dir)
# Train model

model.fit(input_fn=train_input_fn, steps=500)

results = model.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print "%s: %s" % (key, results[key])


# Binarize the output
y_test = dense_to_one_hot(y_test,num_classes=num_classes)

# Compute the confusion matrix
y_predict = np.array(model.predict_proba(input_fn=eval_input_fn))
predicted = np.argmax(y_predict,1)
actual = np.argmax(y_test,1)
accuracy = np.mean(np.equal(predicted,actual))


def evaluate(predicted, actual, num_classes=num_classes):
    confusion = np.zeros((num_classes, num_classes))
    for i in range(len(predicted)):
        for j in range(num_classes):
            if predicted[i] == j:
                for k in range(num_classes):
                    if actual[i] == k:
                        confusion[k][j] += 1
    return confusion

matrix_binary = evaluate(predicted,actual,num_classes=num_classes)

TP = matrix_binary[0][0]
FN = matrix_binary[0][1]
FP = matrix_binary[1][0]
TN = matrix_binary[1][1]
precision_1 = TN/(TN+FN)
recall_1 = TN/(TN + FP)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_predict[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_predict.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


'''
# plot of ROC curve for a specific class
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
'''

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= num_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['green', 'red', 'blue',
                'magenta', 'cyan', 'black', 'silver'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right",fontsize='x-small')
plt.show()