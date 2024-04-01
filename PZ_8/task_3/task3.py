# Імпортуємо необхідні бібліотеки
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Завантажуємо датасет
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
data = pd.read_csv(url, names=names)

# Перевіряємо перші кілька рядків датасету
print(data.head())

# Перетворюємо категоріальні значення в числові
label_encoder = LabelEncoder()
for col in data.columns:
    data[col] = label_encoder.fit_transform(data[col])

# Розділяємо дані на вхідні та вихідні змінні
X = data.drop('class', axis=1)
y = data['class']

# Розділяємо дані на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Класифікація з використанням Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)

# Оцінюємо точність класифікації
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Точність класифікації за допомогою Random Forest:", rf_accuracy)
print(classification_report(y_test, rf_predictions))

# Кластеризація з використанням KMeans та візуалізація з використанням PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)
clusters = kmeans.predict(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis')
plt.title('Кластеризація автомобілів')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

# Передбачення значень з тестової вибірки
predictions = rf_classifier.predict(X_test)

# Створення DataFrame з фактичними та передбаченими значеннями
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})

# Збереження результатів у файл CSV
results_df.to_csv('regression_results.csv', index=False)
