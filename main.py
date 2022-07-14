from dash import Dash, html, dcc, Input, Output
import pygad as pg
import pandas as pd
import numpy as np
import plotly.express as px

function_inputs = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 2, 1, 3, 4, 5, 6, 7, 8, 9],
    [0, 1, 2, 4, 5, 3, 6, 7, 8, 9],
    [1, 9, 3, 5, 7, 2, 8, 4, 6, 0]
]

desired_output = 45


class Pump:
    def __init__(self, capacity, operation_cost):
        self.capacity = capacity
        self.operation_cost = operation_cost


def ff(solution, solution_idx):
    output = np.sum(solution * function_inputs)
    fitness = 1.0 / np.abs(output - desired_output)
    return fitness


def simulation(solution, predicted_states, pumps, delta_time, tunnel_capacity):
    number_of_steps = len(predicted_states)
    number_of_pumps = len(pumps)
    pump_states = solution.reshape(number_of_pumps, number_of_steps)
    calculated_states = [] * number_of_steps
    operational_cost = 0
    overflow = 0
    for step in range(number_of_steps):
        flow = 0
        for pump in range(number_of_pumps):
            operational_cost += pump_states[pump][step] * pumps[pump].operation_cost
            flow += pumps[pump].capacity * pump_states[pump][step] * delta_time
        calculated_states[step] = predicted_states - flow
        # if calculated_states[step] > (tunnel_capacity * 0.95):  # 5% margin of error
        #     if (tunnel_capacity * 0.95) - calculated_states[step] > overflow:
        #         overflow = (tunnel_capacity * 0.95) - calculated_states[step]

    return calculated_states, operational_cost


#   Genes:
#
#   ST1, D1, ST2, D2, ST3, D3, ...
#
#   pump_state_1: [start_time, duration]
#   pump_state_2: [start_time, duration]
#   pump_state_3: [start_time, duration]
#   ...

def solution_converter(solution, number_of_steps, delta_time):
    margin = 0.05  # 5% margin of error
    return


def fitness_function(solution, solution_idx, predicted_states=None, pumps=None, delta_time=0, tunnel_capacity=0):
    sol = solution_converter(solution, len(predicted_states), delta_time)

    margin = 0.05  # 5% margin of error

    calculated_states, operational_cost = simulation(sol, predicted_states, pumps, delta_time, tunnel_capacity)
    number_of_steps = len(calculated_states)
    excess_capacity = [] * number_of_steps
    fitness = number_of_steps / operational_cost
    for step in range(number_of_steps):
        excess_capacity[step] = (tunnel_capacity * (1 - margin)) - calculated_states[step]
        if excess_capacity[step] < 0:
            overflow = -excess_capacity[step]
            fitness /= (1 + overflow / (tunnel_capacity * (1 - margin))) ^ 4


# Load the dataset
avocado = pd.read_csv('avocado-updated-2020.csv')

# Create the Dash app
app = Dash()

# Set up the app layout
geo_dropdown = dcc.Dropdown(options=avocado['geography'].unique(),
                            value='New York')

app.layout = html.Div(children=[
    html.H1(children='Avocado Prices Dashboard'),
    geo_dropdown,
    dcc.Graph(id='price-graph')
])


# Set up the callback function
@app.callback(
    Output(component_id='price-graph', component_property='figure'),
    Input(component_id=geo_dropdown, component_property='value')
)
def update_graph(selected_geography):
    filtered_avocado = avocado[avocado['geography'] == selected_geography]
    line_fig = px.line(filtered_avocado,
                       x='date', y='average_price',
                       color='type',
                       title=f'Avocado Prices in {selected_geography}')
    return line_fig


# Run local server
if __name__ == '__main__':
    app.run_server(debug=True)


    ga_instance = pg.GA(num_generations=5000,
                    num_parents_mating=4,
                    fitness_func=ff,
                    sol_per_pop=50,
                    num_genes=len(function_inputs),
                    init_range_low=-2,
                    init_range_high=2,
                    parent_selection_type="sss",
                    keep_parents=1,
                    crossover_type="single_point",
                    mutation_type="random",
                    mutation_percent_genes=10)

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    prediction = np.sum(np.array(function_inputs) * solution)
    print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

    pumps = [Pump(capacity=100, operation_cost=10), Pump(50, 5), Pump(150, 12)]
    delta_time = 300  # in seconds
    tunnel_capacity = 10 ^ 10  # in gallons
