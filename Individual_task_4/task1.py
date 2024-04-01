import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Визначення діапазонів значень для температури та кута повороту
temperature = np.arange(0, 101, 1)
angle = np.arange(0, 91, 1)

# Визначення функцій належності для температури
temperature_low = fuzz.trimf(temperature, [0, 0, 50])
temperature_medium = fuzz.trimf(temperature, [20, 50, 80])
temperature_high = fuzz.trimf(temperature, [50, 100, 100])

# Визначення функцій належності для кута повороту
angle_small = fuzz.trimf(angle, [0, 0, 45])
angle_medium = fuzz.trimf(angle, [10, 45, 80])
angle_large = fuzz.trimf(angle, [45, 90, 90])

# Візуалізація функцій належності
plt.figure(figsize=(10, 6))
plt.plot(temperature, temperature_low, 'b', linewidth=1.5, label='Low')
plt.plot(temperature, temperature_medium, 'g', linewidth=1.5, label='Medium')
plt.plot(temperature, temperature_high, 'r', linewidth=1.5, label='High')
plt.plot(angle, angle_small, 'b--', linewidth=1.5)
plt.plot(angle, angle_medium, 'g--', linewidth=1.5)
plt.plot(angle, angle_large, 'r--', linewidth=1.5)
plt.xlabel('Temperature')
plt.ylabel('Membership')
plt.title('Membership functions')
plt.legend(['Low', 'Medium', 'High', 'Angle Small', 'Angle Medium', 'Angle Large'])
plt.grid(True)

# Визначення правил для нечіткої системи виведення
rule1 = np.fmax(temperature_low[:, np.newaxis], angle_small)
rule2 = np.fmax(temperature_medium[:, np.newaxis], angle_medium)
rule3 = np.fmax(temperature_high[:, np.newaxis], angle_large)

# Визначення діапазону вхідних даних температури для дослідження
input_temperatures = np.arange(0, 101, 10)
output_angles = []

# Обчислення максимального кута повороту для кожної вхідної температури
for input_temperature in input_temperatures:
    # Знаходимо індекс ближчого значення температури
    index = np.abs(temperature - input_temperature).argmin()
    # Обчислення максимального виходу для кожної температури
    output_angle = np.fmax(np.fmax(rule1, rule2), rule3)[index]
    output_angles.append(output_angle)

# Візуалізація результатів
plt.figure(figsize=(10, 6))
plt.plot(input_temperatures, output_angles, 'ko', markersize=8)
plt.xlabel('Temperature (°C)')
plt.ylabel('Angle (°)')
plt.title('Fuzzy Shower Control System')
plt.grid(True)
plt.show()







