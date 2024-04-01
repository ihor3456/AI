# Імпортуємо необхідні бібліотеки
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Завантажуємо датасет Titanic
data = pd.read_csv('tested.csv')

# Використовуємо метод "one-hot" для кодування категоріальних змінних
data = pd.get_dummies(data, columns=['Sex', 'Embarked'])

# Виконуємо попередню обробку даних
data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Визначаємо вхідні та вихідні змінні
X = data.drop('Survived', axis=1)
y = data['Survived']

# Розділяємо дані на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Визначаємо SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')

# Обробляємо дані тренувальної та тестової вибірок
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Стандартизуємо дані
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Побудова моделі нейронної мережі
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Прогнозуємо класи для тестової вибірки
y_pred = model.predict(X_test)

# Оцінюємо точність моделі
accuracy = accuracy_score(y_test, y_pred)
print("Точність моделі:", accuracy)

# Виводимо матрицю помилок
conf_matrix = confusion_matrix(y_test, y_pred)
print("Матриця помилок:")
print(conf_matrix)

# Візуалізація даних
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

with open('analysis.txt', 'w') as f:
    f.write("Точність моделі: {}\n".format(accuracy))
    f.write("Матриця помилок:\n")
    f.write(np.array2string(conf_matrix, separator=', '))

