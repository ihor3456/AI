import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Завантаження даних
data = pd.read_csv('heart_failure_clinical.csv') 

# Перегляд перших декількох рядків даних
print(data.head())

# Розмір даних
print("Розмір даних:", data.shape)

# Перевірка на наявність пропущених значень
print("Пропущені значення:", data.isnull().sum())

# Розділення даних на ознаки та цільову змінну
X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']

# Розділення даних на тренувальний, валідаційний та тестовий набори
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Стандартизація даних
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# Побудова моделі нейронної мережі
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Компіляція моделі
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Навчання моделі
history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_val_scaled, y_val))


# Оцінка моделі
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print("Точність моделі на тестовому наборі:", accuracy)

# Візуалізація процесу навчання моделі
plt.plot(history.history['accuracy'], label='Точність на тренувальному наборі')
plt.plot(history.history['val_accuracy'], label='Точність на валідаційному наборі')
plt.xlabel('Епохи')
plt.ylabel('Точність')
plt.legend()
plt.show()


# Розрахунок кореляційної матриці
correlation_matrix = data.corr()

# Візуалізація кореляційної матриці
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Кореляційна матриця')
plt.show()



#6-7 task

# Розділення даних на тренувальну та тестову вибірку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Навчання моделі логістичної регресії
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Прогнозування на тестовій вибірці
y_pred = log_reg.predict(X_test)

# Оцінка точності моделі
accuracy = accuracy_score(y_test, y_pred)
print("Точність моделі:", accuracy)

# Виведення класифікаційного звіту
print("Класифікаційний звіт:")
print(classification_report(y_test, y_pred))

# Виведення матриці плутанини
print("Матриця плутанини:")
print(confusion_matrix(y_test, y_pred))


# Візуалізація точності моделі
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='g')
plt.xlabel('Предсказані значення')
plt.ylabel('Справжні значення')
plt.title('Матриця плутанини для моделі логістичної регресії')
plt.show()



# Збереження файлу з аналізом даних та результатами тестування алгоритму машинного навчання
with open('analysis_results.txt', 'w') as file:
    file.write("Аналіз даних та результати тестування алгоритму машинного навчання\n\n")
    file.write("Точність моделі: {}\n\n".format(accuracy))
    file.write("Класифікаційний звіт:\n{}\n\n".format(classification_report(y_test, y_pred)))
    file.write("Матриця плутанини:\n{}\n".format(confusion_matrix(y_test, y_pred)))

