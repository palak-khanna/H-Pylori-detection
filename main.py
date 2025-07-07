import os
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt

# Define paths
base_dir = 'C:\Palak\Research paper\Paper 1\Data\giemsa'
positive_dir = os.path.join(base_dir, 'positive')
negative_dir = os.path.join(base_dir, 'negative')

# List all files
positive_files = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir) if os.path.isfile(os.path.join(positive_dir, f))]
negative_files = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir) if os.path.isfile(os.path.join(negative_dir, f))]

# Split the data
train_pos, test_pos = train_test_split(positive_files, test_size=0.2, random_state=42)
train_neg, test_neg = train_test_split(negative_files, test_size=0.2, random_state=42)

# Create directories for training and testing data
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

train_pos_dir = os.path.join(train_dir, 'positive')
train_neg_dir = os.path.join(train_dir, 'negative')
test_pos_dir = os.path.join(test_dir, 'positive')
test_neg_dir = os.path.join(test_dir, 'negative')

os.makedirs(train_pos_dir, exist_ok=True)
os.makedirs(train_neg_dir, exist_ok=True)
os.makedirs(test_pos_dir, exist_ok=True)
os.makedirs(test_neg_dir, exist_ok=True)

# Copy files to the respective directories
for file in train_pos:
    shutil.copy(file, train_pos_dir)
for file in train_neg:
    shutil.copy(file, train_neg_dir)
for file in test_pos:
    shutil.copy(file, test_pos_dir)
for file in test_neg:
    shutil.copy(file, test_neg_dir)

# ImageDataGenerator for training and testing
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load train and test sets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=3,
    validation_data=test_generator
)

# Predict on the test set
test_generator.reset()
predictions = model.predict(test_generator)
predictions = (predictions > 0.5).astype(int)

# True labels
true_labels = test_generator.classes

# Classification report
report = classification_report(true_labels, predictions, target_names=['negative', 'positive'])
print(report)

# Confusion matrix
cm = confusion_matrix(true_labels, predictions)
print(cm)

# Sensitivity and Specificity
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print(f'Sensitivity: {sensitivity}')
print(f'Specificity: {specificity}')

# AUROC
auc = roc_auc_score(true_labels, predictions)
print(f'AUROC: {auc}')

# ROC Curve
fpr, tpr, thresholds = roc_curve(true_labels, predictions)
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
