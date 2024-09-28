import numpy as np
import tensorflow as tf
# from tabnet import TabNetClassifier
from VT_Forecast.libs.DeAE import models
# from VT_Forecast.libs.VIME.vime_semi import vime_semi


def transform_labels(labels):
    return tf.one_hot(labels, 2)


def preprocess_tabnet_data(X_train, Y_train, X_val, Y_val, col_names, batch_size=50):
    def transform(X, Y):
        features = tf.unstack(X)
        labels = Y

        x = dict(zip(col_names, features))
        y = tf.one_hot(labels, 2)
        return x, y

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train.astype(np.float32), Y_train))
    train_dataset = train_dataset.map(transform)
    train_dataset = train_dataset.batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_val.astype(np.float32), Y_val))
    test_dataset = test_dataset.map(transform)
    test_dataset = test_dataset.batch(batch_size)
    
    return train_dataset, test_dataset


def train_tabnet_classifier(train_dataset, test_dataset, col_names, epochs=20, verbose=2):
    feature_columns = []
    for col_name in col_names:
        feature_columns.append(tf.feature_column.numeric_column(col_name))

    model = TabNetClassifier(
        feature_columns, 
        num_classes=2,
        feature_dim=len(feature_columns), 
        output_dim=2,
        num_decision_steps=10, 
        relaxation_factor=1.0,
        sparsity_coefficient=1e-5,
        batch_momentum=0.98,
        virtual_batch_size=None,
        norm_type='group',
        num_groups=1
    )

    lr = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=100, decay_rate=0.9, staircase=False)
    optimizer = tf.keras.optimizers.Adam(lr)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, verbose=verbose)
    return model


def train_deae_classifier(train_dataset, test_dataset, hidden_features, classes, verbose=False):
    model = models.SdA(train_dataset.shape[1], hidden_features, classes, verbose=verbose).to('cpu')
    model.fit(train_dataset, test_dataset)
    
    return model