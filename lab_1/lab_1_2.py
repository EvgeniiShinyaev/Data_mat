import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)


file_path = 'D:/python/matstat/me/lab_1/Datasets/babyboom.dat.txt'
columns = ["Время_рождения", "Пол", "Вес_рождения", "Минуты_после_полуночи"]
data = pd.read_fwf(file_path, colspecs=[(0, 8), (9, 16), (17, 24), (25, 32)], names=columns)


numeric_data = data[["Вес_рождения", "Минуты_после_полуночи"]]


desc_stats = numeric_data.describe().T
desc_stats["выборочная дисперсия"] = numeric_data.var()



print("Описательная статистика:")
print(desc_stats)


correlations = numeric_data.corr()
print("\nПопарные коэффициенты корреляции:")
print(correlations)


for col in numeric_data.columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Гистограмма
    sns.histplot(numeric_data[col], kde=True, ax=axes[0])
    axes[0].set_title(f'Гистограмма {col}')

    # Ящик с усами
    sns.boxplot(y=numeric_data[col], ax=axes[1])
    axes[1].set_title(f'Ящик с усами для {col}')

    plt.show()
