# Spaceship Titanic — Machine Learning Pipeline

Решение соревнования по предсказанию перемещения пассажиров в другое измерение.

---

## 🛠 Технологии и библиотеки

* **Data Processing & EDA:** `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Feature Engineering & Preprocessing:** `scikit-learn` (`StandardScaler`, `KMeans`, `silhouette_score`)
* **Machine Learning Models:**
* Baseline: `LogisticRegression`, `KNeighborsClassifier`, `SVC`, `RandomForestClassifier`
* Gradient Boosting: `HistGradientBoostingClassifier`, `CatBoostClassifier`, `LightGBM`, `XGBoost`


* **Model Evaluation & Interpretability:** `ConfusionMatrixDisplay`, `SHAP`

---

## 📌 Основные этапы работы

### 1. EDA и визуализация пропусков

* Разработан пользовательский модуль в отдельном скрипте (`.py`) для визуализации пропущенных значений в виде графиков, что позволило выявить паттерны отсутствия данных.
* Проведена очистка данных: обработка пропусков в числовых и категориальных признаках.

### 2. Анализ связей и кодирование

* Построена матрица корреляций (`sns.heatmap` + `.corr()`) для анализа мультиколлинеарности и связи фичей с целевой переменной.
* Выполнено кодирование категориальных переменных с помощью `get_dummies`.

### 3. Feature Engineering

* **Кластеризация K-Means:** С помощью алгоритма `KMeans` сформирован новый признак кластера пассажиров. Оптимальное количество кластеров подобрано на основе оценки силуэта (`silhouette_score`).

### 4. Обучение моделей и валидация

Сравнение работы классических алгоритмов и градиентного бустинга:

* Baseline модели: Logistic Regression, KNN, Support Vector Classifier (SVC), Random Forest.
* Продвинутые ансамбли: HistGradientBoosting, LightGBM, XGBoost, CatBoost.
* Оценка качества матриц ошибок с использованием `ConfusionMatrixDisplay`.

### 5. Интерпретация результатов (SHAP)

* Использован фреймворк `SHAP` (SHapley Additive exPlanations) для анализа важности признаков.

---

## 📁 Структура проекта
```text
├── data/
│   ├── sample_submission.csv  
│   ├── test.csv               
│   └── train.csv              
├── src/
│   ├── modules/
│   │   └── utils.py           
│   ├── Spaceship.ipynb        
│   └── submission.csv         
├── .gitignore                 
└── requirements.txt           

```
