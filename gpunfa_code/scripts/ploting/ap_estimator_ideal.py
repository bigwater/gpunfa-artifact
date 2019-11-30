import pandas as pd
import os

AP_SIZE = 49152

def get_app_and_num_of_state():
    script_path = os.path.dirname(os.path.realpath(__file__))
    #print(script_path)

    df = pd.read_excel(script_path + '/applications.xlsx', sheet_name=0)
    num_of_states = df.loc[0]

    app_and_state = []
    for (app, num_of_state) in zip(num_of_states.index, num_of_states):
        app_and_state.append((app, (num_of_state)))

    app_and_state = app_and_state[1:]
    app_and_state = [(a, int(n)) for (a, n) in app_and_state]

    app_and_state = sorted(app_and_state, key=lambda student: student[1], reverse=True)

    #print(app_and_state)
    return app_and_state


def get_throughput(ap_size, input_size, num_input, ideal=True):
    apps = get_app_and_num_of_state()
    res = {}
    for app, num_of_state in apps:
        #print(app, num_of_state)
        num_offload = num_of_state // ap_size + 1
        
        #print(num_offload)
        cycles = input_size * num_input * num_offload
        cost_time = cycles * 7.5 * 10**(-9) 
        if not ideal:
            cost_time += (num_offload - 1) * 0.05

        throughput = input_size * num_input / cost_time

        res[app] = throughput

    return res


if __name__ == '__main__':
    apps = get_app_and_num_of_state()
    tmp = get_throughput(apps, AP_SIZE, 1000, 1000)
    print(tmp)
    
