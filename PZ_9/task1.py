import numpy as np
import skfuzzy as fuzzy
from skfuzzy import control as ctrl

# Вхідні та вихідні змінні
temperature = ctrl.Antecedent(np.arange(0, 51, 1), 'temperature')
rotation = ctrl.Consequent(np.arange(0, 91, 1), 'rotation')

# Функції належності
temperature['low'] = fuzzy.trimf(temperature.universe, [0, 0, 20])
temperature['medium'] = fuzzy.trimf(temperature.universe, [15, 25, 35])
temperature['high'] = fuzzy.trimf(temperature.universe, [30, 50, 50])

rotation['low'] = fuzzy.trimf(rotation.universe, [0, 0, 30])
rotation['medium'] = fuzzy.trimf(rotation.universe, [20, 40, 60])
rotation['high'] = fuzzy.trimf(rotation.universe, [50, 90, 90])

# Нечіткі правила
rule1 = ctrl.Rule(temperature['low'], rotation['low'])
rule2 = ctrl.Rule(temperature['medium'], rotation['medium'])
rule3 = ctrl.Rule(temperature['high'], rotation['high'])

# Створення системи керування
shower_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
shower = ctrl.ControlSystemSimulation(shower_ctrl)

# Дослідження системи на різних значеннях вхідних даних
shower.input['temperature'] = 25
shower.compute()
print(shower.output['rotation'])

shower.input['temperature'] = 45
shower.compute()
print(shower.output['rotation'])


# Зберегти параметри FIS у файл MATLAB
with open("shower_control.fis", "w") as f:
    f.write("temperature = [0 0 20 15 25 35 30 50 50];\n")
    f.write("rotation = [0 0 30 20 40 60 50 90 90];\n")
    f.write("ruleList = [1 1 0 0 0; 2 2 0 0 0; 3 3 0 0 0];\n")

print("Fuzzy Inference System parameters saved to shower_control.fis.")