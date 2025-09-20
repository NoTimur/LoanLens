import numpy as np
import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def delete_outliers(data, column, quantile):
    upper_bound = data[column].quantile(quantile)
    data[column] = data[column].clip(upper=upper_bound)

    return data

def fill_missing_values(data, column, method='mean'):
    if method == 'mean':
        fill_value = data[column].mean()
    elif method == 'median':
        fill_value = data[column].median()
    else:
        raise ValueError("Method not supported")
    data[column].fillna(fill_value, inplace=True)

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Генерация новых признаков на основе существующих"""
    df = df.copy()

    # Соотношение долга к доходу
    df['Debt_to_Income'] = df['DebtRatio'] * df['MonthlyIncome']

    # Общее количество просрочек
    df['TotalPastDue'] = (
        df['PastDueLess_60'] + df['PastDue60_90'] + df['PastDue90_More']
    )

    # Индикатор наличия просрочек
    df['HasPastDue'] = (df['TotalPastDue'] > 0).astype(int)

    # Среднее количество просрочек на кредит
    df['PastDue_per_Loan'] = df['TotalPastDue'] / (df['OpenLoans'] + 1)

    # Доход на члена семьи
    df['Income_per_FamilyMember'] = df['MonthlyIncome'] / (df['FamilySize'] + 1)

    # Баланс на открытый кредит
    df['Balance_per_Loan'] = df['TotalBalanceDivideCreditLimits'] / (df['OpenLoans'] + 1)

    # Наличие кредита на недвижимость
    df['HasRealEstateLoan'] = (df['RealEstateLoans'] > 0).astype(int)

    # Отношение открытых кредитов к недвижимости
    df['OpenLoan_to_RealEstateLoan'] = df['OpenLoans'] / (df['RealEstateLoans'] + 1)

    # Инверсные признаки
    df['Income_inverse'] = 1 / (df['MonthlyIncome'] + 1)
    df['Age_inverse'] = 1 / (df['Age'] + 1)

    return df

def log_transform(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Применяет логарифмическое преобразование к числовым признакам"""
    df = df.copy()
    for col in cols:
        df[col] = np.log1p(df[col])
    return df
