import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def table_to_numeric(df: pd.DataFrame, id_cols=None, target_cols=None,  max_categories=6) -> pd.DataFrame:
    """ Трансформация датафрейма: категориальные столбцы бинаризируются.
    Не рекомендуется использовать этот обработчик многоразово, т.к. некоторые преобразования
    (кодирование категорий, масштабирование) зависят от данных.
    """
    # колонка с ID пользователя
    id_cols = id_cols or []
    # целевая колонка
    target_cols = target_cols or []
    # категорические колонки
    cat_cols = df.nunique()[df.nunique() < max_categories].keys().tolist()
    cat_cols = [x for x in cat_cols if x not in target_cols]
    # числовые колонки
    num_cols = [x for x in df.columns if x not in cat_cols + target_cols + id_cols]
    # бинарные колонки с двумя значениями
    bin_cols = df.nunique()[df.nunique() == 2].keys().tolist()
    # колонки с более, чем двумя значениями
    multi_cols = [i for i in cat_cols if i not in bin_cols]

    print('categorical columns:', multi_cols)
    print('binary columns:', bin_cols)
    print('numeric columns:', num_cols)

    # кодировщик лейблов для бинарных колонок
    le = LabelEncoder()
    for i in bin_cols:
        df[i] = le.fit_transform(df[i])

    # дублируем для многозначных колонок
    df = pd.get_dummies(data=df, columns=multi_cols)

    # расширяем числовые колонки
    std = StandardScaler()
    scaled = std.fit_transform(df[num_cols])
    scaled = pd.DataFrame(scaled, columns=num_cols)

    # сбрасываем предыдущие значения с объед.
    df = df.drop(columns=num_cols, axis=1)
    df = df.merge(scaled, left_index=True, right_index=True, how="left")
    return df
