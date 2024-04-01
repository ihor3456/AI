import numpy as np

# Задана логічна функція (X1 OR X2) AND X3 AND X4
# Створимо таблицю істинності
truth_table = np.array([[0, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0],
                        [0, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 1],
                        [0, 1, 1, 0],
                        [0, 1, 1, 1],
                        [1, 0, 0, 0],
                        [1, 0, 0, 1],
                        [1, 0, 1, 0],
                        [1, 0, 1, 1],
                        [1, 1, 0, 0],
                        [1, 1, 0, 1],
                        [1, 1, 1, 0],
                        [1, 1, 1, 1]])

# Побудова моделі багатошарової нейронної мережі без використання TensorFlow
class SimpleNeuralNetwork:
    def __init__(self):
        # Ініціалізація випадкових ваг та зсувів для нейронів
        self.w1 = np.random.randn(3, 2)  # Ваги для першого шару (3 входи, 2 нейрони)
        self.b1 = np.zeros((1, 2))       # Зсув для першого шару
        self.w2 = np.random.randn(2, 1)  # Ваги для другого шару (2 входи з першого шару, 1 вихід)
        self.b2 = 0                      # Зсув для другого шару

    # Реалізація функції активації ReLU
    def relu(self, x):
        return np.maximum(0, x)

    # Реалізація функції активації Sigmoid
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Функція передбачення, яка виконує пряме поширення
    def predict(self, x):
        z1 = np.dot(x, self.w1) + self.b1  # Лінійне перетворення для першого шару
        a1 = self.relu(z1)                 # Активація першого шару з використанням ReLU
        z2 = np.dot(a1, self.w2) + self.b2  # Лінійне перетворення для другого шару
        a2 = self.sigmoid(z2)              # Активація другого шару з використанням Sigmoid
        return a2

# Створення та навчання простої нейронної мережі
simple_model = SimpleNeuralNetwork()
X_train = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y_train = np.array([0, 1, 1, 0, 1, 0, 0, 1])

# Процес навчання простої нейронної мережі (1000 ітерацій)
for _ in range(1000):
    for i in range(len(X_train)):
        x = X_train[i].reshape(1, -1)  # Перетворення вхідних даних до необхідного розміру
        y = y_train[i]                 # Очікуване вихідне значення для навчання

        # Пряме поширення: обчислення вихідного значення за допомогою поточних ваг та зсувів
        z1 = np.dot(x, simple_model.w1) + simple_model.b1
        a1 = simple_model.relu(z1)
        z2 = np.dot(a1, simple_model.w2) + simple_model.b2
        a2 = simple_model.sigmoid(z2)

        # Зворотнє поширення: обчислення похідних та оновлення ваг та зсувів з використанням градієнтного спуску
        dz2 = a2 - y
        dw2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2)
        da1 = np.dot(dz2, simple_model.w2.T)
        dz1 = da1 * (a1 > 0)
        dw1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0)

        # Оновлення ваг та зсувів з використанням швидкості навчання 0.01
        simple_model.w1 -= 0.01 * dw1
        simple_model.b1 -= 0.01 * db1
        simple_model.w2 -= 0.01 * dw2
        simple_model.b2 -= 0.01 * db2

# Тестування простої нейронної мережі
test_input = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
for i in range(len(test_input)):
    x_test = test_input[i].reshape(1, -1)
    prediction = simple_model.predict(x_test)
    print("Prediction for input {}: {}".format(test_input[i], prediction))

# Аналіз результатів навчання простої нейронної мережі
def binary_cross_entropy(y_true, y_pred):
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Обчислення середнього значення функції втрат для всіх навчальних прикладів
total_loss = 0
for i in range(len(X_train)):
    x_train = X_train[i].reshape(1, -1)
    prediction = simple_model.predict(x_train)
    loss = binary_cross_entropy(y_train[i], prediction)
    total_loss += loss

# Виведення середнього значення втрат
average_loss = total_loss / len(X_train)
print("Average Loss: {}".format(average_loss))
