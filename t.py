import pandas as pd

# Загрузка результатов теста
results_df = pd.read_csv("classification_results.csv")  # или используй results_df из ModelTester

# Подсчет количества каждого класса
class_counts = results_df["Label"].value_counts()
print(class_counts)
