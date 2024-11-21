import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Загрузка данных
file_path = 'D:/python/matstat/me/lab_2/Datasets/iris.txt'
data = pd.read_csv(file_path, sep=',', header=None)
data.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]


# Функция для проверки нормальности и построения доверительных интервалов
def analyze_distribution(data, label):
    # Проверка нормальности (Shapiro-Wilk test)
    stat, p_value = stats.shapiro(data)
    print(f"\nПроверка нормальности для {label}:")
    print(f"Статистика Shapiro-Wilk = {stat:.4f}, p-значение = {p_value:.4f}")

    # Интерпретация результата теста Шапиро-Уилка
    if p_value > 0.05:
        print(f"Не отвергаем нулевую гипотезу: распределение длины чашелистика '{label}' возможно нормальное.")
    else:
        print(f"Отвергаем нулевую гипотезу: распределение длины чашелистика '{label}' не является нормальным.")

    # Доверительные интервалы для среднего и стандартного отклонения
    mean = np.mean(data)
    std_err = stats.sem(data)
    conf_int_mean = stats.norm.interval(0.95, loc=mean, scale=std_err)

    std_dev = np.std(data, ddof=1)
    conf_int_std = (std_dev * np.sqrt(len(data) - 1) / stats.chi2.ppf(0.975, len(data) - 1),
                    std_dev * np.sqrt(len(data) - 1) / stats.chi2.ppf(0.025, len(data) - 1))

    print(f"Среднее: {mean:.4f}, Доверительный интервал для среднего (95%): {conf_int_mean}")
    print(
        f"Стандартное отклонение: {std_dev:.4f}, Доверительный интервал для стандартного отклонения (95%): {conf_int_std}")


# 1. Анализ для каждого типа ириса
for iris_type in data['class'].unique():
    iris_data = data[data['class'] == iris_type]["sepal_length"]
    print(f"\nАнализ для класса {iris_type}:")
    analyze_distribution(iris_data, f"{iris_type}")
