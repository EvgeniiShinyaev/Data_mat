import pandas as pd
import scipy.stats as stats

# Загрузка данных
file_path = 'D:/python/matstat/me/lab_3/Datasets/babyboom.dat.txt'
columns = ['Time_of_birth', 'Sex', 'Birth_weight', 'Minutes_after_midnight']
data = pd.read_csv(file_path, sep='\s+', names=columns)

# Разделение данных по полу
girls_weight = data[data['Sex'] == 1]['Birth_weight']
boys_weight = data[data['Sex'] == 2]['Birth_weight']

stat, p = stats.shapiro(girls_weight)
print(stat, p)
if p > 0.05:
    print('распределение нормальное')
else:
    print('распределение ненормальное')

#Т.к распределение девочек ненормальное
# Гипотеза 1: Проверка, равен ли средний вес девочек среднему весу мальчиков
t_stat, p_value_mean = stats.ttest_ind(boys_weight, girls_weight, equal_var=False)
print(f"Проверка среднего веса: t-статистика = {t_stat}, p-значение = {p_value_mean}")

# Гипотеза 2: Проверка, равна ли дисперсия веса девочек дисперсии веса мальчиков
f_stat, p_value_var = stats.levene(girls_weight, boys_weight)
print(f"Проверка дисперсии: F-статистика = {f_stat}, p-значение = {p_value_var}")

# Интерпретация
if p_value_mean < 0.05:
    print("Отвергаем нулевую гипотезу для средних весов: Есть значительная разница в среднем весе между девочками и мальчиками.")
else:
    print("Не отвергаем нулевую гипотезу для средних весов: Нет значительной разницы в среднем весе между девочками и мальчиками.")

if p_value_var < 0.05:
    print("Отвергаем нулевую гипотезу для дисперсий: Есть значительная разница в дисперсии веса между девочками и мальчиками.")
else:
    print("Не отвергаем нулевую гипотезу для дисперсий: Нет значительной разницы в дисперсии веса между девочками и мальчиками.")
