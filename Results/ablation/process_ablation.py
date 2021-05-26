import random

import pandas as pd
import plotly.graph_objects as go

if __name__ == '__main__':
    for f in ["ablation-simverb.csv","ablation-simlex.csv","ablation-card.csv"]:
        file = f
        df = pd.read_csv(file)
        skip = ["MIN", "MAX", "Step"]
        models = []
        exam = ""
        for i in df.columns:
            should_skip = False
            for skippable in skip:
                if skippable in i:
                    should_skip = True
                    break
            if should_skip: continue
            models.append(i)
            exam = i.split("-")[1]

        # Add data

        full_fig = go.Figure()
        oov_fig = go.Figure()
        dash = ['dash', 'dot', 'dashdot']
        colors = ['red', 'green', 'blue', 'purple', 'orange', 'pink']
        markers = ["circle", "square", "diamond", "cross", "star", "triangle-up"]
        models = sorted(models)
        name_mapping = {
            "alllosses": "All Losses",
            "nocyclemae": "-cycle_mae_loss  -cycle_maxmargin_loss -cycle_discriminator_loss",
            "nocycledis": "-cycle_discriminator_loss",
            "noonewaymm": "-cycle_mae_loss -cycle_maxmargin_loss -cycle_discriminator_loss -id_loss -one_way_maxmargin_loss",
            "nocycleid": "-cycle_mae_loss  -cycle_maxmargin_loss  -cycle_discriminator_loss  -id_loss",
            "nocyclemm": "-cycle_maxmargin_loss -cycle_discriminator_loss"
        }
        for model in models:
            marker = markers[models.index(model) % 6]
            color = colors[models.index(model)%6]
            # Create and style traces
            fig = None
            if "full" in model:
                fig = full_fig
            else:
                fig = oov_fig
            name = model.split("-")[0]
            loss_mapping = name_mapping[name.split("_")[-1].strip()]
            fig.add_trace(
                go.Scatter(x=df["Step"], y=df[model], name=loss_mapping, mode='lines+markers', marker_symbol=marker,
                           line=dict(color=color, width=1)
                           )
            )

        # Edit the layout
        full_fig.update_layout(title='Full Scenario Ablation Tests',
                               xaxis_title='Steps',
                               yaxis_title=exam + " (Spearman Rho)",
                               legend=dict(
                                   yanchor="auto",
                                   xanchor="auto",
                                   x=0,
                                   y=0,
                                   bgcolor="rgba(0,0,0,0)"
                               )
                               )
        oov_fig.update_layout(title='Disjoint Scenario Ablation Tests',
                              xaxis_title='Steps',
                              yaxis_title=exam + " (Spearman Rho)",
                              legend=dict(
                                  yanchor="auto",
                                  xanchor="auto",
                                  x=0,
                                  y=0,
                                  bgcolor="rgba(0,0,0,0)"
                              ))

        full_fig.show()
        oov_fig.show()

    for f in ["ablation-toggle-simverb.csv","ablation-toggle-simlex.csv","ablation-toggle-card.csv"]:
        file = f
        df = pd.read_csv(file)
        skip = ["MIN", "MAX", "Step"]
        models = []
        exam = ""
        for i in df.columns:
            should_skip = False
            for skippable in skip:
                if skippable in i:
                    should_skip = True
                    break
            if should_skip: continue
            models.append(i)
            exam = i.split("-")[1]

        # Add data

        full_fig = go.Figure()
        oov_fig = go.Figure()
        dash = ['dash', 'dot', 'dashdot']
        colors = ['red', 'green', 'blue', 'purple', 'orange', 'pink']
        markers = ["circle", "square", "diamond", "cross", "star", "triangle-up"]
        models = sorted(models)
        name_mapping = {
            "alllosses": "All Losses",
            "nocyclemae": "-cycle_mae_loss",
            "nocycledis": "-cycle_discriminator_loss",
            "noonewaymm": "-one_way_maxmargin_loss",
            "nocycleid": "-id_loss",
            "nocyclemm": "-cycle_maxmargin_loss"
        }
        for model in models:
            marker = markers[models.index(model) % 6]
            color = colors[models.index(model)%6]
            # Create and style traces
            fig = None
            if "full" in model:
                fig = full_fig
            else:
                fig = oov_fig
            name = model.split("-")[0]
            loss_mapping = name_mapping[name.split("_")[-1].strip()]
            fig.add_trace(
                go.Scatter(x=df["Step"], y=df[model], name=loss_mapping, mode='lines+markers', marker_symbol=marker,
                           line=dict(color=color, width=1)
                           )
            )

        # Edit the layout
        full_fig.update_layout(title='Full Scenario Toggle Ablation Tests',
                               xaxis_title='Steps',
                               yaxis_title=exam + " (Spearman Rho)",
                               legend=dict(
                                   yanchor="auto",
                                   xanchor="auto",
                                   x=0,
                                   y=0,
                                   bgcolor="rgba(0,0,0,0)"
                               )
                               )
        oov_fig.update_layout(title='Disjoint Scenario Toggle Ablation Tests',
                              xaxis_title='Steps',
                              yaxis_title=exam + " (Spearman Rho)",
                              legend=dict(
                                  yanchor="auto",
                                  xanchor="auto",
                                  x=0,
                                  y=0,
                                  bgcolor="rgba(0,0,0,0)"
                              ))

        full_fig.show()
        oov_fig.show()
