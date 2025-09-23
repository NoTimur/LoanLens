import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def filter_family_size(df: pd.DataFrame, max_size: int = 10) -> pd.DataFrame:
    return df[df['FamilySize'] <= max_size]

def delete_outliers(df, columns, quantile):
    df = df.copy()
    for column in columns:
        upper_bound = df[column].quantile(quantile)
        df[column] = df[column].clip(upper=upper_bound)
    return df


def fill_missing_values(df, columns, method='median'):
    df = df.copy()
    for col in columns:
        if method == 'mean':
            fill_val = df[col].mean()
        elif method == 'median':
            fill_val = df[col].median()
        else:
            raise ValueError(f"Unsupported method: {method}")
        df[col] = df[col].fillna(fill_val)
    return df

def impute_with_regression(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Заполняет пропуски в target_col с помощью линейной регрессии.

    Пошагово:
    - обучается модель на строках без пропусков
    - заполняет пропуски предсказанными значениями
    - удаляет строки с минимальными и максимальными значениями target_col

    Parameters:
    - df: входной DataFrame
    - target_col: имя колонки с пропущенными значениями

    Returns:
    - df с заполненными значениями в target_col
    """
    df = df.copy()

    # Разделяем данные
    train = df[df[target_col].notna()]
    test = df[df[target_col].isna()]

    if test.empty:
        return df  # нечего заполнять

    # Подготовка признаков
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])

    # Числовые признаки — заполняем медианой
    for col in X_train.select_dtypes(include=np.number).columns:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    # One-hot кодирование категориальных признаков
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)

    # Приводим X_test к тем же колонкам, что и X_train
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Обучаем модель
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Предсказываем пропущенные значения
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, a_min=0, a_max=None)  # убираем отрицательные

    # Обновляем значения в df
    df.loc[test.index, target_col] = y_pred

    # Удаляем экстремальные значения
    min_val = df[target_col].min()
    max_val = df[target_col].max()
    df = df[(df[target_col] != min_val) & (df[target_col] != max_val)]

    return df

    
def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Безопасная генерация признаков
    def safe_divide(a, b):
        return np.where(b == 0, 0, a / b)

    try:
        df['Debt_to_Income'] = df['DebtRatio'] * df['MonthlyIncome']
        df['TotalPastDue'] = df['PastDueLess_60'] + df['PastDue60_90'] + df['PastDue90_More']
        df['HasPastDue'] = (df['TotalPastDue'] > 0).astype(int)
        df['PastDue_per_Loan'] = safe_divide(df['TotalPastDue'], df['OpenLoans'] + 1)
        df['Income_per_FamilyMember'] = safe_divide(df['MonthlyIncome'], df['FamilySize'] + 1)
        df['Balance_per_Loan'] = safe_divide(df['TotalBalanceDivideCreditLimits'], df['OpenLoans'] + 1)
        df['HasRealEstateLoan'] = (df['RealEstateLoans'] > 0).astype(int)
        df['OpenLoan_to_RealEstateLoan'] = safe_divide(df['OpenLoans'], df['RealEstateLoans'] + 1)
        df['Income_inverse'] = 1 / (df['MonthlyIncome'] + 1)
        df['Age_inverse'] = 1 / (df['Age'] + 1)
    except KeyError as e:
        print(f"❌ Column missing: {e}. Проверь входные данные.")
        raise

    return df
