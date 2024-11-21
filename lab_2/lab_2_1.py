import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Загрузка данных
file_path = 'D:/python/matstat/me/lab_2/Datasets/babyboom.dat.txt'
columns = ["Time_of_birth", "Sex", "Birth_weight", "Minutes_after_midnight"]
data = pd.read_fwf(file_path, colspecs=[(0, 8), (9, 16), (17, 24), (25, 32)], names=columns)


# 1. Проверка нормальности веса младенцев
def check_normality(data, label):
    # Проверка нормальности (Shapiro-Wilk test)
    stat, p_value = stats.shapiro(data)
    print(f"\nПроверка нормальности для {label}:")
    print(f"Статистика Shapiro-Wilk test: {stat}, p-значение: {p_value}")


    if p_value > 0.05:
        print("Не отвергаем нулевую гипотезу.  Вероятно, распределение веса младенцев - нормальное.")
    else:
        print("Отвергаем нулевую гипотезу.  Вероятно, распределение веса младенцев не является нормальным.")
    return p_value


# Проверка для всех детей
p_value_all = check_normality(data["Birth_weight"], "Вес всех младенцев")

# Проверка для девочек и мальчиков по отдельности
girls_weight = data[data["Sex"] == 1]["Birth_weight"]
boys_weight = data[data["Sex"] == 2]["Birth_weight"]

p_value_girls = check_normality(girls_weight, "Вес девочек")
p_value_boys = check_normality(boys_weight, "Вес мальчиков")

print()

# Доверительные интервалы для параметров нормального распределения
def confidence_interval_normal(data, alpha=0.05):
    mean = np.mean(data)
    std_err = stats.sem(data)
    conf_int = stats.norm.interval(1 - alpha, loc=mean, scale=std_err)
    #print(f"\nДоверительный интервал: {conf_int} при уровне значимости {alpha}")
    return conf_int


print(f"Доверительный интервал для веса всех младенцев:\n"
      f"{confidence_interval_normal(data['Birth_weight'])}")
#confidence_interval_normal(data["Birth_weight"])
print()
print(f"Доверительный интервал для веса девочек:\n"
      f"{confidence_interval_normal(girls_weight)}")
#confidence_interval_normal(girls_weight)
print()
print(f"Доверительный интервал для веса мальчиков:\n"
      f"{confidence_interval_normal(boys_weight)}")

# 2. Проверка гипотезы об экспоненциальном распределении времени между рождениями
birth_intervals = np.diff(data["Minutes_after_midnight"])
#print(birth_intervals)
print("\nПроверка гипотезы о распределении времени между рождениями:")

# Оценка параметра λ
lambda_est = 1 / np.mean(birth_intervals)
print(f"Оценка параметра λ: {lambda_est}")

# Проверка на экспоненциальное распределение с помощью критерия Колмогорова-Смирнова
D, p_value_exp = stats.kstest(birth_intervals, 'expon', args=(0, lambda_est))
print(f"Статистика Колмогорова-Смирнова: {D}, p-значение: {p_value_exp}")

if p_value_exp < 0.05:
    print('Отвергаем нулевую гипотезу.  Вероятно, время между рождением детей не подчиняется экспоненциальному распределению.')
else:
    print('Не отвергаем нулевую гипотезу.  Вероятно, время между рождением детей подчиняется экспоненциальному распределению.')


# 3. Проверка гипотезы о распределении Пуассона для количества рождений в час
data['Hour'] = data['Minutes_after_midnight'] // 60
births_per_hour = data['Hour'].value_counts().sort_index()

# Оценка параметра λ для Пуассоновского распределения
lambda_poisson = births_per_hour.mean()
print("\nПроверка гипотезы о распределении Пуассона для количества рождений в час:")
print(f"Оценка параметра λ: {lambda_poisson}")

# Проверка с помощью критерия Хи-квадрат
observed_freq = births_per_hour.values
expected_freq = np.mean(observed_freq) * np.ones_like(observed_freq)
chi2_stat, p_value_poisson = stats.chisquare(f_obs=observed_freq, f_exp=expected_freq)
print(f"Статистика Хи-квадрат: {chi2_stat}, p-значение: {p_value_poisson}")

if p_value_poisson > 0.05:
    print("Не отвергаем нулевую гипотезу.  Вероятно, количество рождений в час для каждого часа подчиняется распределению Пуассона.")
else:
    print("Отвергаем нулевую гипотезу.  Вероятно, количество рождений в час для каждого часа не подчиняется распределению Пуассона.")

