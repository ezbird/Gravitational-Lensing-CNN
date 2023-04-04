'''
Test CNN for COSMOS 2020 gravitational lensing 
detection.

April 2023
Ezra Huscher
'''

import tensorflow as tf

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

train_dir = '/home/dobby/LensingProject/augmented_training_set'
validation_dir = '/home/dobby/LensingProject/training_set'
input_shape = (32, 32, 3)
num_classes = 2

BATCH_SIZE = 1                                              # 32
IMG_SIZE = (32, 32)

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=False,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)
                                                            
validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=False,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)
                                                            
class_names = train_dataset.class_names

print('Number of training batches: %d' % tf.data.experimental.cardinality(train_dataset).numpy())
print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset).numpy())

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
DataSet
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Initialize
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(5, 5), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)
model.add(tf.keras.layers.Flatten( input_shape=(32,32,3) ) )
model.add(tf.keras.layers.Dense(2))
'''
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=( 32, 32, 3 )),
    tf.keras.layers.Reshape((32, 32 * 3)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM( 32, return_sequences=True, return_state=False )),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM( 32 )),
    tf.keras.layers.Dense( 2 ),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense( 2 ),

])

model.add(tf.keras.layers.Flatten( input_shape=(32,32,3) ) )
model.add(tf.keras.layers.Dense(2))
'''
model.summary()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Optimizer
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

optimizer = tf.keras.optimizers.Nadam(
    learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
    name='Nadam'
) # 0.00001


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Loss Fn
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""                               
# 1
# lossfn = tf.keras.losses.MeanSquaredLogarithmicError(reduction=tf.keras.losses.Reduction.AUTO, name='mean_squared_logarithmic_error')

# 2
lossfn = tf.keras.losses.SparseCategoricalCrossentropy( from_logits=False, reduction=tf.keras.losses.Reduction.AUTO, name='sparse_categorical_crossentropy')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Summary
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model.compile( loss=lossfn, metrics=['accuracy', tf.keras.metrics.CategoricalAccuracy()]) #optimizer=optimizer,

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Training
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
history = model.fit(train_dataset, epochs=15 ,validation_data=(validation_dataset))
    
input("Press Any Key!")