# включение файлов в директории "../input/"
import os
# import matplotlib.pyplot as plt #для визуализации
from PIL import Image

import numpy as np
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objects as go
import plotly.tools as tls
import plotly.figure_factory as ff
# from IPython.display import display


# для исключения проблемной отрисовки в colab
def configure_plotly_browser_state():
    import IPython
    # the display function is imported by the colab notebook
    display(IPython.core.display.HTML(
        '''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',
            },
          });
        </script>
        '''
    ))


def plot_pie_chart(column):
    vc = column.value_counts()
    lab = vc.keys().tolist()
    val = vc.values.tolist()

    trace = go.Pie(labels=lab,
                   values=val,
                   marker=dict(colors=['royalblue', 'lime'],
                               line=dict(color="white",
                                         width=1.3)
                               ),
                   rotation=90,
                   hoverinfo="label+value+text",
                   hole=.5,
                   )
    layout = go.Layout(dict(
        # title = "Churn rate",
        plot_bgcolor="rgb(243,243,243)",
        paper_bgcolor="rgb(243,243,243)",
        annotations=[
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Доля оттока",

            }
        ]
    )
    )
    data = [trace]
    fig = go.Figure(data=data, layout=layout)

    configure_plotly_browser_state()
    fig.show()


def plot_correlation(df):
    configure_plotly_browser_state()

    # корреляция
    correlation = df.corr()

    # обозначаем лейблы
    matrix_cols = correlation.columns.tolist()

    # конвертируем в массив
    corr_array = np.array(correlation)

    # рисуем
    trace = go.Heatmap(z=corr_array,
                       x=matrix_cols,
                       y=matrix_cols,
                       colorscale="Viridis",
                       colorbar=dict(title="Коэффициент корреляции Пирсона",
                                     titleside="right"
                                     ),
                       )

    layout = go.Layout(dict(annotations=[
        {
            "font": {
                "size": 16
            },
            "showarrow": False,
            "text": "Корреляционная матрица",
            "x": 20,
            "y": 35
        }
    ],
        autosize=False,
        height=720,
        width=800,
        margin=dict(r=0, l=210,
                    t=25, b=210,
                    ),
        yaxis=dict(tickfont=dict(size=9)),
        xaxis=dict(tickfont=dict(size=9))
    )
    )

    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)
