from math import pi

import pandas as pd
from collections import Counter
import random
from bokeh.models import (HoverTool, FactorRange, Plot, LinearAxis, Grid,
                          Range1d)
from bokeh.models.glyphs import VBar
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models.sources import ColumnDataSource, Dict
from flask import Flask, render_template
from bokeh.palettes import Greens256, Category20c
from bokeh.plotting import figure, show
from bokeh.transform import cumsum

def get_chart(labels):
    x = {
            'Hate': 0,
            'Offensive': 0,
            'None': 0
    }
    for l in labels:
        if l in x.keys():
                x[l] += 1
        else:
                x[l] = 1

    data = pd.DataFrame.from_dict(dict(x), orient='index').reset_index()
    data = data.rename(index=str, columns={0:'value', 'index':'label'})
    data['angle'] = data['value']/sum(x.values()) * 2*pi
    data['color'] = Category20c[len(x)]
    p = figure(plot_height=350, toolbar_location=None,
                tools="hover", tooltips=[("Label", "@label"),("Value", "@value")])
    p.wedge(x=0, y=1, radius=0.4, 
                start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                line_color="white", fill_color='color', legend='label', source=data)

    p.axis.axis_label=None
    p.axis.visible=False
    p.grid.grid_line_color = None

    return components(p)
