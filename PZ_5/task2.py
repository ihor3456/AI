import numpy as np

# Згенеруємо випадкові дані про автомобілі
np.random.seed(0)
num_samples = 1000

# Рік випуску (від 2000 до 2022)
year = np.random.randint(2000, 2023, size=num_samples)

# Кількість циліндрів (від 4 до 12)
cylinders = np.random.randint(4, 13, size=num_samples)

# Потужність двигуна (від 100 до 500 к.с.)
horsepower = np.random.randint(100, 501, size=num_samples)

# Вартість (від 10000 до 100000)
price = np.random.randint(10000, 100001, size=num_samples)

# Клас автомобіля: 0 - економ, 1 - стандарт, 2 - преміум
car_class = np.random.randint(0, 3, size=num_samples)

# Перевірка перших 5 записів
print("Перші 5 записів:")
print("Рік випуску | Кількість циліндрів | Потужність двигуна | Вартість | Клас")
for i in range(5):
    print(f"{year[i]}           | {cylinders[i]}                     | {horsepower[i]}                      | {price[i]}   | {car_class[i]}")

# Нормалізація числових ознак
def normalize_feature(feature):
    return (feature - np.min(feature)) / (np.max(feature) - np.min(feature))

# Нормалізація кожної числової ознаки
year = normalize_feature(year)
cylinders = normalize_feature(cylinders)
horsepower = normalize_feature(horsepower)
price = normalize_feature(price)

# Перевірка перших 5 записів
print("\nПерші 5 записів після нормалізації:")
print("Рік випуску | Кількість циліндрів | Потужність двигуна | Вартість | Клас")
for i in range(5):
    print(f"{year[i]}           | {cylinders[i]}                     | {horsepower[i]}                      | {price[i]}   | {car_class[i]}")



class NeuralNetwork:
    def __init__( self,input_size, hidden_size, output_size):
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

# Перевірка роботи моделі
input_size = 4 # кількість параметрів автомобіля
hidden_size = 5 # кількість прихованих нейронів
output_size = 3 # кільк




4. # Навчання моделі


# Розбиття даних на тренувальний та тестовий набори
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    np.array([year, cylinders, horsepower, price]).T,  # вхідні дані
    car_class,  # вихідні дані
    test_size=0.2,
    random_state=42
)

# Навчання моделі
epochs = 1000
model = NeuralNetwork(input_size, hidden_size, output_size)  # Replace input_size, hidden_size, and output_size with the appropriate values
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


# Матриця плутанини
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, predictions_test)
print("Матриця плутанини:")
print(conf_matrix)

# Інші метрики
from sklearn.metrics import classification_report

class_names = ["Економ", "Стандарт", "Преміум"]
print("Звіт про класифікацію:")
print(classification_report(y_test, predictions_test, target_names=class_names))
