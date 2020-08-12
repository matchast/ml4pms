import pandas as pd

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from yellowbrick.classifier import DiscriminationThreshold

from ml4pms.visualization import go, py, tls


def train_and_evaluate_classifier(
        algorithm,
        training_x, testing_x, training_y, testing_y,
        cols,
        cf='coefficients',
        threshold_plot=True,
):
    """
    algorithm     - использованный алгоритм с методами fit, predict и predict_proba
    training_x    - данные для предсказывающих переменных (обучение)
    testing_x     - данные для предсказывающих переменных (тест)
    training_y    - целевая переменная (обучение)
    training_y    - целевая переменная (тест)
    cf - ["coefficients","features"](коэффициенты для логрегрессии, параметры для деревьев)
    threshold_plot - если True, возвращает для модели threshold plot
    """

    # модель
    algorithm.fit(training_x, training_y)
    predictions = algorithm.predict(testing_x)
    probabilities = algorithm.predict_proba(testing_x)

    # коэффициенты
    if cf == "coefficients":
        coefficients = pd.DataFrame(algorithm.coef_.ravel())
    elif cf == "features":
        coefficients = pd.DataFrame(algorithm.feature_importances_)
    else:
        raise ValueError('`coefficients` value must be one of {`coefficients`, `features`}')

    column_df = pd.DataFrame(cols)
    coef_sumry = (pd.merge(coefficients, column_df, left_index=True, right_index=True, how="left"))
    coef_sumry.columns = ["coefficients", "features"]
    coef_sumry = coef_sumry.sort_values(by="coefficients", ascending=False)

    print(algorithm)
    print("\n Отчет по классфицикации : \n", classification_report(testing_y, predictions))
    print("Точность : ", accuracy_score(testing_y, predictions))
    # confusion_matrix
    conf_matrix = confusion_matrix(testing_y, predictions)
    # roc_auc_score
    model_roc_auc = roc_auc_score(testing_y, predictions)
    print("Площадь под кривой : ", model_roc_auc, "\n")
    fpr, tpr, thresholds = roc_curve(testing_y, probabilities[:, 1])

    # готовим confusion_matrix для рисования
    trace1 = go.Heatmap(z=conf_matrix,
                        x=["Пользователи", "Отток"],
                        y=["Пользователи", "Отток"],
                        showscale=False, colorscale="Picnic",
                        name="matrix",
                        )

    # готовим roc_curve для рисования
    trace2 = go.Scatter(x=fpr, y=tpr,
                        name="Roc : " + str(model_roc_auc),
                        line=dict(color=('rgb(22, 96, 167)'), width=2))
    trace3 = go.Scatter(x=[0, 1], y=[0, 1],
                        line=dict(color=('rgb(205, 12, 24)'), width=2,
                                  dash='dot'))

    # готовим коэффициенты для рисования
    trace4 = go.Bar(x=coef_sumry["features"], y=coef_sumry["coefficients"],
                    name="coefficients",
                    marker=dict(color=coef_sumry["coefficients"],
                                colorscale="Picnic",
                                line=dict(width=.6, color="black")))

    # рисуем
    fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]])

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)
    fig.append_trace(trace3, 1, 2)
    fig.append_trace(trace4, 2, 1)

    fig['layout'].update(showlegend=False,
                         autosize=False, height=900, width=800,
                         plot_bgcolor='rgba(240,240,240, 0.95)',
                         paper_bgcolor='rgba(240,240,240, 0.95)',
                         margin=dict(b=195),
                         )
    fig["layout"]["xaxis2"].update(dict(title="false positive rate"))
    fig["layout"]["yaxis2"].update(dict(title="true positive rate"))
    fig["layout"]["xaxis3"].update(dict(showgrid=True, tickfont=dict(size=10),
                                        tickangle=90))
    py.iplot(fig)

    if threshold_plot:
        visualizer = DiscriminationThreshold(algorithm)
        visualizer.fit(training_x, training_y)
        visualizer.show()
