import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Загрузка данных
file_path = 'D:/python/matstat/me/lab_2/Datasets/euroweight.dat.txt'

data = pd.read_csv(file_path,sep='\t',header=None)
data.columns = ["ID", "weight", "batch"]

# Преобразование столбца weight в числовой формат
data["weight"] = pd.to_numeric(data["weight"], errors='coerce')

# Удаление строк с некорректными значениями в weight
data = data.dropna(subset=["weight"])


# Функция для проверки нормальности и построения доверительных интервалов
def analyze_distribution(data, label):
    # Проверка нормальности (Shapiro-Wilk test)
    stat, p_value = stats.shapiro(data)
    print(f"\nПроверка нормальности для {label}:")
    print(f"Статистика Shapiro-Wilk = {stat:.4f}, p-значение = {p_value:.4f}")

    # Интерпретация результата теста Шапиро-Уилка
    if p_value > 0.05:
        print(f"Не отвергаем нулевую гипотезу: распределение веса монет в '{label}' возможно нормальное.")
    else:
        print(f"Отвергаем нулевую гипотезу: распределение веса монет в '{label}' не является нормальным.")

    # Доверительные интервалы для среднего и стандартного отклонения
    mean = np.mean(data)
    std_err = stats.sem(data)
    conf_int_mean = stats.norm.interval(0.95, loc=mean, scale=std_err)

    print(f"Среднее: {mean:.4f}, Доверительный интервал для среднего: {conf_int_mean}")


# 1. Анализ для всей выборки
print("Анализ для всех монет:")
analyze_distribution(data["weight"], "всех монет")

# 2. Анализ для каждого пакета отдельно
for batch_num in sorted(data['batch'].unique()):
    batch_data = data[data['batch'] == batch_num]["weight"]
    print(f"\nАнализ для пакета {batch_num}:")
    analyze_distribution(batch_data, f"пакета {batch_num}")
