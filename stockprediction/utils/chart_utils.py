def get_layout() -> dict:
    """ Method to get the layout used for Plotly charts """
    layout = {
            "xaxis": {"rangeselector": {
                "x": 0,
                "y": 0.9,
                "font": {"size": 13},

                "buttons": [
                    {
                        "step": "all",
                        "count": 1,
                        "label": "reset"
                    },
                    {
                        "step": "month",
                        "count": 3,
                        "label": "3 mo",
                        "stepmode": "backward"
                    },
                    {
                        "step": "month",
                        "count": 1,
                        "label": "1 mo",
                        "stepmode": "backward"
                    },
                    {"step": "all"}
                ]
            }},
            "yaxis": {
                "domain": [0, 0.2],
                "showticklabels": False,
            },
            "legend": {
                "x": 0.3,
                "y": 0.9,
                "yanchor": "bottom",
                "orientation": "h"
            },
            "margin": {
                "b": 30,
                "l": 30,
                "r": 30,
                "t": 30,
            },
            "yaxis2": {"domain": [0.2, 0.8]},
            "plot_bgcolor": "rgb(250, 250, 250)"
        }

    return layout
