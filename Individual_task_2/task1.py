import math

def calculate_circle_area(radius):
    area = math.pi * radius**2
    return area

def main():
    # Запит користувача на введення радіуса кола
    radius = float(input("Будь ласка, введіть радіус кола: "))
    
    # Обчислення площі кола за допомогою функції
    area = calculate_circle_area(radius)
    
    # Виведення результату
    print("Площа кола з радіусом", radius, "дорівнює", area)

if __name__ == "__main__":
    main()
