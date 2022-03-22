from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from covid_model import *

def agent_portrayal(agent):
    portrayal = {
        'Shape': 'circle',
        'Layer': 0,
        'r': 0.6,
        'Color': '#66F'}

    # (Un)masked agents show up as (non-)filled circles
    # if agent.masked == True:
    #     portrayal['Filled'] = 'true'

    if agent.infected == True:
        portrayal['Color'] = '#F66'

    if agent.immune == True:
        portrayal['Color'] = '#6C6'

    if agent.lockdown == True:
        portrayal['Filled'] = 'true'

    return portrayal

grid = CanvasGrid(agent_portrayal, 70, 50, 700, 500)

line_charts = ChartModule(series = [
    {'Label': '易感人群', 'Color': '#66F', 'Filled': 'false'}, 
    {'Label': '感染人群', 'Color': '#F66', 'Filled': 'false'},
    {'Label': '死亡人群', 'Color': 'black', 'Filled': 'false'},
    {'Label': '痊愈及免疫人群', 'Color': '#6C6', 'Filled': 'false'}])

server = ModularServer(CovidModel,
                       [grid, line_charts],
                       '新冠肺炎传播模拟',
                       model_params)

server.port = 8521  # default port if unspecified
server.launch()