import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np

# Завантаження даних MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормалізація даних
x_train, x_test = x_train / 255.0, x_test / 255.0

# Вибірка 30 зразків для тренування
random_indices = np.random.choice(x_train.shape[0], 30, replace=False)
x_train = x_train[random_indices]
y_train = y_train[random_indices]

# Оголошення архітектури нейронної мережі
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Компіляція моделі
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Тренування моделі
history = model.fit(x_train, y_train, epochs=3, batch_size=30, validation_split=0.1)

# Оцінка точності на тестових даних
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Точність на тестових даних:", test_acc)

# Перевірка розпізнавання вказаних цифр
digit_indices = []
for digit in [5, 25, 7]:
    indices = np.where(y_test == digit)[0]
    if len(indices) > 0:
        digit_indices.append(indices[0])
    else:
        print(f"Цифра {digit} не знайдена у тестовому наборі даних.")

digit_images = x_test[digit_indices]
predictions = model.predict(digit_images)
predicted_labels = np.argmax(predictions, axis=1)
print("Помилково розпізнані зразки:")
mistakes_count = 0
for i, (true_label, predicted_label) in enumerate(zip([5, 25, 7], predicted_labels)):
    if true_label != predicted_label:
        print(f"Зразок {i+1}: Справжнє значення: {true_label}, Розпізнане значення: {predicted_label}")
        mistakes_count += 1

print("Кількість помилково розпізнаних зразків:", mistakes_count)

# Запис результатів точності навчання
with open('training_accuracy.txt', 'w') as f:
    f.write('Accuracy: {}\n'.format(history.history['accuracy']))
    f.write('Validation Accuracy: {}\n'.format(history.history['val_accuracy']))

# Збереження файлу з результатами розпізнавання
with open('recognition_results.txt', 'w') as f:
    f.write('True Label\tPredicted Label\n')
    for true_label, predicted_label in zip([5, 25, 7], predicted_labels):
        f.write('{}\t{}\n'.format(true_label, predicted_label))



