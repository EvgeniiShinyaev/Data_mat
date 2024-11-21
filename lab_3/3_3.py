import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
import statsmodels.api as sm

# Загрузка данных
file_path = 'D:/python/matstat/me/lab_2/Datasets/iris.txt'
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'flower_class']
data = pd.read_csv(file_path, names=columns)

# Проверка гипотез о равенстве распределений для каждой характеристики с помощью теста Краскела-Уоллиса
print("Тест Краскела-Уоллиса для равенства распределений характеристик:")
for feature in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
    groups = [data[data['flower_class'] == flower_type][feature] for flower_type in data['flower_class'].unique()]
    h_stat, p_value = stats.kruskal(*groups)
    print(f"{feature}: H-статистика = {h_stat}, p-значение = {p_value}")
    if p_value < 0.05:
        print(f"  Распределения значений {feature} различаются между классами.")
    else:
        print(f"  Распределения значений {feature} не различаются между классами.")

# Проверка гипотезы о равенстве средних значений для каждой характеристики с помощью ANOVA
print("\nANOVA для проверки равенства средних значений:")
for feature in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
    model = ols(f'{feature} ~ C(flower_class)', data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(f"{feature}:\n", anova_table)
    if anova_table['PR(>F)'].iloc[0] < 0.05:
        print(f"  Средние значения {feature} значимо различаются между классами.")
    else:
        print(f"  Средние значения {feature} не различаются между классами.")

# Проверка гипотезы о равенстве дисперсий для каждой характеристики с помощью теста Левена
print("\nТест Левена для проверки равенства дисперсий:")
for feature in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
    groups = [data[data['flower_class'] == flower_type][feature] for flower_type in data['flower_class'].unique()]
    w_stat, p_value = stats.levene(*groups)
    print(f"{feature}: W-статистика = {w_stat}, p-значение = {p_value}")
    if p_value < 0.05:
        print(f"  Дисперсии значений {feature} значимо различаются между классами.")
    else:
        print(f"  Дисперсии значений {feature} не различаются между классами.")
