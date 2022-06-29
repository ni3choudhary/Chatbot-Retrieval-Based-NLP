# Importing Required Libraries
import pickle
from nltk.stem import WordNetLemmatizer
import random
import numpy as np
import tensorflow as tf

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Unpickling Files
classes = pickle.load(open('classes.pkl', 'rb'))
documents = pickle.load(open('documents.pkl', 'rb'))
words = pickle.load(open('words.pkl', 'rb'))

# Initialize Empty list
training = []
output_empty = [0] * len(classes)
# ignore punctuation from words
ignore_words = ['?',',','!','.','-']

for doc in documents:
    pattern_words = doc[0]  # patterns
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words if word not in ignore_words]

    # Create a Bag for Each Pattern
    bag = [ 1 if w in pattern_words else 0 for w in words ]

    # Classes
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    # Add X = Bag and y = Classes to Training List
    training.append([bag, output_row])

# Shuffle Data to Avoid Overfitting
random.shuffle(training)
training = np.array(training)

# Split Training Data into X_train and y_train 
X_train = list(training[:, 0])
y_train = list(training[:, 1])

# Model Building : Define the model
model = tf.keras.Sequential()
# Optionally, the first layer can receive an 'input_shape' argument..
model.add(tf.keras.layers.Dense(128,
            input_shape=(len(X_train[0]),),
            activation = 'relu' ))
# Dropout Layer 
model.add(tf.keras.layers.Dropout(0.5))
# Afterwards, we do automatic shape inference..
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
# Dropout Layer 
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(y_train[0]), activation = 'softmax'))

# Define Optimizer 
opt = tf.keras.optimizers.SGD(learning_rate=0.01,
                              nesterov=True,
                              momentum=0.9,
                              decay = 1e-6)

# Compile the Model
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics= ['accuracy'])    

# Fit the Model
model.fit(np.array(X_train), np.array(y_train), epochs=200, batch_size=32, verbose=0)

# Save the Model
model.save('chatbot_model.h5') 
print('Model Saved Successfully!!!')                     