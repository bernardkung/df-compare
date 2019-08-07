import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table

import plotly
import plotly.graph_objs as go

import requests
import json
from sodapy import Socrata

import pandas as pd
import datetime as dt
from itertools import chain

from sklearn import datasets, linear_model
import statsmodels.api as sm


## Version using the sodapy package to query the socrata data API
# Unauthenticated client only works with public data sets. 
# Requests made without an app_token will be subject to strict throttling limits.
client = Socrata('data.medicare.gov',
                 'UXvDXkqXFr9NqLO4q2A9cvoeP',
                #  userame="user@example.com",
                #  password="AFakePassword",
)

results = client.get("eqxu-aw4f")
fivestar = pd.DataFrame.from_records(results)

predictor_columns = ['standard_infection_ratio',
                     'standardized_hospitalization_ratio',
                     'standardized_readmission_ratio',
                     'mortality_rate_facility',
                     'readmission_rate_facility',
                     'fistula_rate_facility']

fivestar[predictor_columns].astype('float', inplace=True)

def data_to_plotly(x):
    k = []
    
    for i in range(0, len(x)):
        k.append(float(x[i][0]))
        
    return k

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([
        dcc.Dropdown(
            id='chain-options',
            options = [
                {'label': c, 'value': c} for c in 
                list(
                    fivestar['chain_organization'] \
                        .value_counts() \
                        .reset_index() \
                        ['index']
                    )],
                value=list(
                    fivestar['chain_organization'] \
                        .value_counts() \
                        .reset_index() \
                        .nlargest(2, 'chain_organization')['index']),
            multi=True,
        ),
    ], className='three columns'),
    html.Div([
        # dropdown row
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='left-options',
                    options=[
                        {'label': p, 'value': p} for 
                            p in predictor_columns
                    ],
                    value='standard_infection_ratio',
                    )
            ], className='six columns'),
            html.Div([
                dcc.Dropdown(
                    id='right-options',
                    options=[
                        {'label': p, 'value': p} for 
                            p in predictor_columns
                    ],
                    value='standardized_hospitalization_ratio',
                    )
            ], className='six columns'),
        ], className='row'),
        # corr graph
        html.Div([
            dcc.Graph(id='corr-graph')
        ], className='row'),
        html.Div(
            style={'margin-top': '100px', 'margin-bottom': '100px'},
            children=[
            html.H6("OLS Regression Results"),
            html.Table([
                html.Tr([
                    html.Td(
                        style={'font-weight':'bold'}, 
                        children=['Dep. Variable']), 
                    html.Td(id='dep-var'),
                    html.Td(
                        style={'font-weight':'bold'}, 
                        children=['No. Observations']), 
                    html.Td(id='num-obs'),
                    html.Td(
                        style={'font-weight':'bold'}, 
                        children=['R-squared']), 
                    html.Td(id='r-sq'),
                    html.Td(
                        style={'font-weight':'bold'}, 
                        children=['F-statistic']), 
                    html.Td(id='f-stat'),
                    html.Td(
                        style={'font-weight':'bold'}, 
                        children=['Log-Likelihood']), 
                    html.Td(id='log-like'),
                    ]),
                html.Tr([
                    html.Td(
                        style={'font-weight':'bold'}, 
                        children=['Model']), 
                    html.Td(id='model'),
                    html.Td(
                        style={'font-weight':'bold'}, 
                        children=['Df Residuals']), 
                    html.Td(id='df-res'),
                    html.Td(
                        style={'font-weight':'bold'}, 
                        children=['Adj. R-squared']), 
                    html.Td(id='adj-r'),
                    html.Td(
                        style={'font-weight':'bold'}, 
                        children=['Prob (F-statistic)']), 
                    html.Td(id='p-fstat'),
                    html.Td(
                        style={'font-weight':'bold'}, 
                        children=['AIC']), 
                    html.Td(id='aic'),
                ]),
                html.Tr([
                    html.Td(
                        style={'font-weight':'bold'}, 
                        children=['Method']), 
                    html.Td(id='method'),
                    html.Td(
                        style={'font-weight':'bold'}, 
                        children=['Df Model']), 
                    html.Td(id='df-mod'),
                    html.Td(), html.Td(),
                    html.Td([]), html.Td(),
                    html.Td(
                        style={'font-weight':'bold'}, 
                        children=['BIC']), 
                    html.Td(id='bic'),
                ]),
            ]),
        ], className='row'),
        # stats row
        html.Div([
            html.Div(
                style={'text-align':'center'},
                children=[
                    html.H6(id='l-stat-title'),
                    html.Table(                        
                        style={'text-align':'center',
                               'margin-left':'auto',
                               'margin-right':'auto',
                               },
                        children=[
                        html.Tr(
                            [html.Td(
                                style={'font-weight':'bold'}, 
                                children=['Count']), 
                            html.Td(id='l-count')]),
                        html.Tr(
                            [html.Td(
                                style={'font-weight':'bold'}, 
                                children=['Mean']), 
                            html.Td(id='l-mean')]),
                        html.Tr(
                            [html.Td(
                                style={'font-weight':'bold'}, 
                                children=['StDev']), 
                            html.Td(id='l-stdev')]),
                        html.Tr(
                            [html.Td(
                                style={'font-weight':'bold'}, 
                                children=['Min']), 
                            html.Td(id='l-min')]),
                        html.Tr(
                            [html.Td(
                                style={'font-weight':'bold'}, 
                                children=['25%']), 
                            html.Td(id='l-q25')]),
                        html.Tr(
                            [html.Td(
                                style={'font-weight':'bold'}, 
                                children=['Med']), 
                            html.Td(id='l-med')]),
                        html.Tr(
                            [html.Td(
                                style={'font-weight':'bold'}, 
                                children=['75%']), 
                            html.Td(id='l-q75')]),
                        html.Tr(
                            [html.Td(
                                style={'font-weight':'bold'}, 
                                children=['Max']), 
                            html.Td(id='l-max')]),
                    ]),
                ], className='six columns'),
            html.Div(                
                style={'text-align':'center'},
                children=[
                    html.H6(id='r-stat-title'),
                    html.Table(
                        style={'text-align':'center',
                               'margin-left':'auto',
                               'margin-right':'auto',
                               },
                        children=[
                        html.Tr([
                            html.Td(
                                style={'font-weight':'bold'}, 
                                children=['Count']),
                            html.Td(id='r-count')]),
                        html.Tr(
                            [html.Td(
                                style={'font-weight':'bold'}, 
                                children=['Mean']), 
                            html.Td(id='r-mean')]),
                        html.Tr(
                            [html.Td(
                                style={'font-weight':'bold'}, 
                                children=['StDev']), 
                            html.Td(id='r-stdev')]),
                        html.Tr(
                            [html.Td(
                                style={'font-weight':'bold'}, 
                                children=['Min']), 
                            html.Td(id='r-min')]),
                        html.Tr(
                            [html.Td(
                                style={'font-weight':'bold'}, 
                                children=['25%']), 
                            html.Td(id='r-q25')]),
                        html.Tr(
                            [html.Td(
                                style={'font-weight':'bold'}, 
                                children=['Med']), 
                            html.Td(id='r-med')]),
                        html.Tr(
                            [html.Td(
                                style={'font-weight':'bold'}, 
                                children=['75%']), 
                            html.Td(id='r-q75')]),
                        html.Tr(
                            [html.Td(
                                style={'font-weight':'bold'}, 
                                children=['Max']), 
                            html.Td(id='r-max')]),
                    ]),
                ], className='six columns'),
        ], className='row'),
    ], className='nine columns'),
], className='container')

@app.callback([
        Output('corr-graph', 'figure'),
        Output('l-count', 'children'),
        Output('l-mean', 'children'),
        Output('l-stdev', 'children'),
        Output('l-min', 'children'),
        Output('l-q25', 'children'),
        Output('l-med', 'children'),
        Output('l-q75', 'children'),
        Output('l-max', 'children'),
        Output('r-count', 'children'),
        Output('r-mean', 'children'),
        Output('r-stdev', 'children'),
        Output('r-min', 'children'),
        Output('r-q25', 'children'),
        Output('r-med', 'children'),
        Output('r-q75', 'children'),
        Output('r-max', 'children'),
        Output('dep-var', 'children'),
        Output('num-obs', 'children'),
        Output('r-sq', 'children'),
        Output('f-stat', 'children'),
        Output('log-like', 'children'),
        Output('model', 'children'),
        Output('df-res', 'children'),
        Output('adj-r', 'children'),
        Output('p-fstat', 'children'),
        Output('aic', 'children'),
        Output('method', 'children'),
        Output('df-mod', 'children'),
        Output('bic', 'children'),
    ],[
        Input('chain-options', 'value'),
        Input('left-options', 'value'),
        Input('right-options', 'value'),
    ])
def update_graphs(selected_chains, selected_left, selected_right):

    ffs = fivestar[
        fivestar['chain_organization'].isin(selected_chains)]

    corr_traces = []
    for c in selected_chains:
        ffs_corr = ffs[ffs['chain_organization']==c]
        if ffs_corr.shape[0] != 0:
            corr_traces.append({
                'x': ffs_corr[selected_left],
                'y': ffs_corr[selected_right],
                'mode': 'markers',
                'type': 'scatter',
                'name': c,
                'text': ffs_corr['facility_name'],
            })

    ffs_linreg = ffs[[selected_left, selected_right]].dropna(how='any')
    
    ffs_corr_x = ffs_linreg[selected_left].values.reshape(-1, 1)
    ffs_corr_y = ffs_linreg[selected_right].values.reshape(-1, 1)

    linreg = linear_model.LinearRegression()
    linreg.fit(ffs_corr_x, ffs_corr_y)

    linreg_trace = {
        'x': data_to_plotly(ffs_corr_x),
        'y': data_to_plotly(linreg.predict(ffs_corr_x)),
        'mode': 'lines',
        'type': 'scatter',
        'name': 'Linear Regression',
    }

    corr_traces.append(linreg_trace)

    corr_layout = {
        'xaxis': {
            'title': selected_left,
        },
        'yaxis': {
            'title': selected_right,
        },
        'margin': {'l': 40, 'b': 40, 't': 10, 'r': 0},
        'hovermode': 'closest',
        'legend':{'x':0, 'y': -0.2, 'orientation': 'h'},
    }

    # statsmodel package provides much better statistics
    # note the ffs_corr_x and ffx_corr_y are arrays of strings
    # data_to_plotly() these to lists of floats
    linreg_model = sm.OLS(data_to_plotly(ffs_corr_y), data_to_plotly(ffs_corr_x)).fit()
    # linreg_predictions = linreg_model.predict(data_to_plotly(ffs_corr_x)) # make the predictions by the model

    # summary is stored in three 'simpleTables'
    # I'm going to take the first table, and convert it to a dictionary 
    # since I only need specific packages.
    # The simpleTable looks like a list of lists with whitespace formatting
    # 1. completely unlist everything into flat list
    # 2. strip white spaces, nulls, and colons
    # 3. zip every two list items into a dictionary key-value pair
    linreg_summary0 = list(
        chain.from_iterable(linreg_model.summary().tables[0].data))
    linreg_summary0 = [x.strip().replace(":", "") 
        for x in linreg_summary0 if x.strip()!='']
    
    linreg_summary = dict(zip(
        linreg_summary0[0::2], 
        linreg_summary0[1::2]))

    linreg_outstats = ['Dep. Variable', 
                       'No. Observations',
                       'R-squared',
                       'F-statistic',
                       'Log-Likelihood',
                       'Model',
                       'Df Residuals',
                       'Adj. R-squared',
                       'Prob (F-statistic)',
                       'AIC',
                       'Method',
                       'Df Model',
                       'BIC',
                      ]

    # need to return a list of dictionary and values, but have a dictionary and two nested lists
    # the nested lists are chain unlisted into a single flat list first
    # dictionary for graph data is then inserted
    # linreg stats are then appended on
    # could replace insert with append by moving output order in app.callback decorator

    # add left and right selected features stats to output
    output_list = list(chain.from_iterable([
            ffs[[selected_left]].astype('float').describe()[selected_left].tolist(),
            ffs[[selected_right]].astype('float').describe()[selected_right].tolist(),
    ]))

    # add corr-graph fig data to output
    output_list.insert(0, {'data': corr_traces, 'layout': corr_layout})

    # add LinReg stats to output
    for stat in linreg_outstats:
        output_list.append(linreg_summary[stat])


    return output_list



@app.callback([
    Output('l-stat-title', 'children'),
    Output('r-stat-title', 'children'),
],[
    Input('left-options', 'value'),
    Input('right-options', 'value'),
])
def update_stat_titles(selected_left, selected_right):
    '''Just passes selected options directly to title headers'''
    return selected_left, selected_right



if __name__ == '__main__':
    app.run_server(debug=True)
