import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def percent_missing(df, column = None, method = "text"):
    if column is not None:
        val = np.round(df[column].isna().mean() * 100, 2)
        if method == "text":
            return f"Процент пропущенных значений в '{column}': {val}%"
        elif method == "num":
            return val
        else:
            raise ValueError("Параметр `method` должен быть либо `text`, либо `num`")

    return np.round(df.isna().mean() * 100, 2)

def visualization_missing(df, ax=None, title=""):
    per_missing = percent_missing(df)
    per_missing = per_missing[per_missing > 0].sort_values()

    if per_missing.empty:
        ax.text(
            0.5,
            0.5,
            "Нет пропущенных значений",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_title(title)
        return
    
    sns.barplot(
        x=per_missing.index,
        y=per_missing,
        hue=per_missing.index,
        palette="tab10",
        legend=False,
        ax=ax,
    )

    ax.set_xlabel("Признаки")
    ax.set_ylabel("Процент пропусков (%)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=90)


def show_missing_data(df_train, df_test):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    visualization_missing(df_train, ax=axes[0], title="Обучающая выборка")
    visualization_missing(df_test, ax=axes[1], title="Тестовая выборка")

    plt.tight_layout()  
    plt.show()

