import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)


file_path = 'D:/python/matstat/me/lab_1/Datasets/airportdat.txt'
columns = ["Airport", "City", "Scheduled departures", "Performed departures", "Enplaned passengers", "Enplaned revenue tons of freight",
        "Enplaned revenue tons of mail"]

data = pd.read_fwf(file_path, colspecs=[(0, 20), (21, 42), (43, 49), (50, 56), (57, 65), (66, 75), (76, 85)],
                   names=columns)


numeric_data = data[["Scheduled departures", "Performed departures", "Enplaned passengers", "Enplaned revenue tons of freight",
        "Enplaned revenue tons of mail"]]


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
