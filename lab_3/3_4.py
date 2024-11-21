import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

# Загрузка данных
data = pd.read_excel("D:/python/matstat/me/lab_3/Datasets//surgery.xlsx", skiprows=1)

print("Названия столбцов:", data.columns)

# Определение успешных случаев
# Определение успешных случаев
data['success'] = (data['V right.1'] > data['V right']) & \
                  (data['V left.1'] > data['V left'])

# Подсчет числа успешных операций и общего количества операций
success_count = data['success'].sum()
total_count = len(data)

# Параметры гипотезы
# Проверяем гипотезу с вероятностью успеха 0.7
p_hypothesis_0_7 = 0.7
stat_0_7, p_value_0_7 = proportions_ztest(success_count, total_count, value=p_hypothesis_0_7)

# Проверяем гипотезу с вероятностью успеха 0.8
p_hypothesis_0_8 = 0.8
stat_0_8, p_value_0_8 = proportions_ztest(success_count, total_count, value=p_hypothesis_0_8)

# Вывод результатов
print(f"Гипотеза с вероятностью успеха 0.7: Z-статистика = {stat_0_7}, p-значение = {p_value_0_7}")
if p_value_0_7 < 0.05:
    print("Отклоняем гипотезу о вероятности успеха 0.7.")
else:
    print("Не отклоняем гипотезу о вероятности успеха 0.7.")

print(f"\nГипотеза с вероятностью успеха 0.8: Z-статистика = {stat_0_8}, p-значение = {p_value_0_8}")
if p_value_0_8 < 0.05:
    print("Отклоняем гипотезу о вероятности успеха 0.8.")
else:
    print("Не отклоняем гипотезу о вероятности успеха 0.8.")
