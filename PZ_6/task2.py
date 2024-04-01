import numpy as np

# Згенеруємо випадкові дані про тварин для прикладу
np.random.seed(0)
num_samples = 1000

# Розмір тварин (від 1 до 10)
size = np.random.randint(1, 11, size=num_samples)

# Колір тварин: 0 - коричневий, 1 - чорний, 2 - білий
color = np.random.randint(0, 3, size=num_samples)

# Тип тварини: 0 - кіт, 1 - собака, 2 - птах
animal_type = np.random.randint(0, 3, size=num_samples)

# Перевірка перших 5 записів
print("Перші 5 записів:")
print("Розмір | Колір | Тип")
for i in range(5):
    print(f"{size[i]}       | {color[i]}     | {animal_type[i]}")

# Нормалізація числових ознак
def normalize_feature(feature):
    return (feature - np.min(feature)) / (np.max(feature) - np.min(feature))

# Нормалізація розміру та конвертація кольору в one-hot encoding
size = normalize_feature(size)
color_one_hot = np.zeros((num_samples, 3))
color_one_hot[np.arange(num_samples), color] = 1

# Перевірка перших 5 записів після нормалізації
print("\nПерші 5 записів після нормалізації:")
print("Розмір | Колір (Коричневий, Чорний, Білий) | Тип")
for i in range(5):
    print(f"{size[i]}       | {color_one_hot[i]}                       | {animal_type[i]}")

# Використання класу нейронної мережі для сегментації тварин
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Ініціалізація ваг та зміщень
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    # Сигмоїдна функція активації
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Пряме поширення
    def forward(self, X):
        self.hidden_sum = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.activated_hidden = self.sigmoid(self.hidden_sum)

        self.output_sum = np.dot(self.activated_hidden, self.weights_hidden_output) + self.bias_output
        self.activated_output = self.sigmoid(self.output_sum)

        return self.activated_output

    # Зворотне поширення
    def backward(self, X, y, output, learning_rate=0.01):
        self.output_error = y - output
        self.output_delta = self.output_error * (output * (1 - output))

        self.hidden_error = np.dot(self.output_delta, self.weights_hidden_output.T)
        self.hidden_delta = self.hidden_error * (self.activated_hidden * (1 - self.activated_hidden))

        self.weights_input_hidden += learning_rate * np.dot(X.T, self.hidden_delta)
        self.weights_hidden_output += learning_rate * np.dot(self.activated_hidden.T, self.output_delta)

# Побудова та навчання моделі
input_size = 4 # кількість параметрів (розмір, колір, тип)
hidden_size = 5 # кількість прихованих нейронів
output_size = 3 # кількість класів тварин

# Розбиття даних на тренувальний та тестовий набори
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    np.concatenate((size.reshape(-1, 1), color_one_hot), axis=1),  # вхідні дані
    animal_type,  # вихідні дані
    test_size=0.2,
    random_state=42
)

# Навчання моделі
epochs = 1000
model = NeuralNetwork(input_size, hidden_size, output_size)
output = model.forward(X_train)
for epoch in range(epochs):
    # Пряме поширення та зворотне поширення для кожного навчального прикладу
    for i in range(len(X_train)):
        X = X_train[i].reshape(1, -1)  # вхідний приклад
        y = np.zeros((1, output_size))  # очікуваний вихід
        y[0, y_train[i]] = 1
        output = model.forward(X_train)  # пряме поширення
        model.backward(X_train, y, output)  # зворотне поширення

    # Оцінка точності моделі після кожної епохи
    if (epoch + 1) % 100 == 0:
        predictions = np.argmax(model.forward(X_train), axis=1)
        accuracy = np.mean(predictions == y_train)
        print(f"Епоха {epoch + 1}/{epochs}, Точність: {accuracy:.4f}")

# Тестування моделі
predictions_test = np.argmax(model.forward(X_test), axis=1)
accuracy_test = np.mean(predictions_test == y_test)
print(f"Точність на тестовому наборі: {accuracy_test:.4f}")

# Обчислення середньоквадратичної функції втрат
loss = np.mean(np.square(y_test - predictions_test))
print(f"Середньоквадратична функція втрат: {loss:.4f}")

# Виведення вагових коефіцієнтів
print("\nВагові коефіцієнти:")
print("Ваги від входу до прихованого шару:")
print(model.weights_input_hidden)
print("\nВаги від прихованого до вихідного шару:")
print(model.weights_hidden_output)
