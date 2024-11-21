import pandas as pd
import itertools
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Загрузка данных
file_path = 'D:/python/matstat/me/lab_3/Datasets/euroweight.dat.txt'
columns = ['ID', 'weight', 'batch']
data = pd.read_csv(file_path, sep='\s+', names=columns)

# Проверка на нормальность распределения

normal_batches = []

for batch in data["batch"].unique():
    print(f"Пакет {batch}: ", end="")
    stat, p = stats.shapiro(data[data["batch"] == batch]["weight"])

    if p > 0.05:
        print("нормальное распределение")
        normal_batches.append(batch)
    else:
        print("не нормальное распределение")
print()
print(len(normal_batches))
print(len(data["batch"].unique()))
# Проверка гипотезы о равенстве средних значений веса монет во всех пакетах (ANOVA или Краскела–Уоллиса)
batches = [data[data["batch"] == batch]["weight"] for batch in data["batch"].unique()]

# Проверяем равенство дисперсий для всех пакетов (тест Левена)
levene_stat, levene_p = stats.levene(*batches)

if len(normal_batches) == len(data["batch"].unique()) and levene_p > 0.05:
    # Все пакеты нормальные и имеют равные дисперсии — используем ANOVA
    anova_result = stats.f_oneway(*batches)
    print("Результаты ANOVA:", anova_result)
    if anova_result.pvalue < 0.05:
        print("Есть статистически значимые различия между средними значениями весов пакетов.")
        # Попарные сравнения с помощью теста Тьюки
        tukey_result = pairwise_tukeyhsd(data["weight"], data["batch"])
        print("Результаты попарного теста Тьюки:\n", tukey_result)
else:
    # Либо есть пакеты с ненормальным распределением, либо дисперсии не равны
    # Используем тест Краскела–Уоллиса как аналог ANOVA для сравнения всех пакетов
    kruskal_result = stats.kruskal(*batches)
    print("Результаты теста Краскела–Уоллиса:", kruskal_result)
    if kruskal_result.pvalue < 0.05:
        print("Есть статистически значимые различия между средними значениями весов пакетов.")
        # Попарные сравнения тестом Манна–Уитни для пакетов, у которых хотя бы один не нормальный или неравные дисперсии
        print("\nПопарные сравнения:")
        pairs = itertools.combinations(data["batch"].unique(), 2)
        for batch1, batch2 in pairs:
            data1 = data[data["batch"] == batch1]["weight"]
            data2 = data[data["batch"] == batch2]["weight"]

            if batch1 in normal_batches and batch2 in normal_batches:
                # Если обе выборки нормальные
                levene_stat, levene_p = stats.levene(data1, data2)
                if levene_p > 0.05:
                    # Используем стандартный t-тест при равных дисперсиях
                    ttest_result = stats.ttest_ind(data1, data2, equal_var=True)
                    print(f"Пакет {batch1} vs Пакет {batch2} (t-тест): p-value = {ttest_result.pvalue}")
                else:
                    # Используем тест Уэлча при неравных дисперсиях
                    welch_result = stats.ttest_ind(data1, data2, equal_var=False)
                    print(f"Пакет {batch1} vs Пакет {batch2} (тест Уэлча): p-value = {welch_result.pvalue}")
            else:
                # Если хотя бы одна из выборок ненормальная — используем тест Манна–Уитни
                mannwhitney_result = stats.mannwhitneyu(data1, data2, alternative="two-sided")
                print(f"Пакет {batch1} vs Пакет {batch2} (тест Манна–Уитни): p-value = {mannwhitney_result.pvalue}")
