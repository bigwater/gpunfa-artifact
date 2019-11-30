import matplotlib
matplotlib.use('agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
from functools import reduce
import os


geomean = lambda n: reduce(lambda x, y: x*y, n) ** (1.0 / len(n))

def get_app_order():
    script_path = os.path.dirname(os.path.realpath(__file__))
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


def readFromFile(filename, kw):
    file_data = {}
    f = open(filename, 'r')

    app_header = None

    index = 0
    for nn, line in enumerate(f):
        line = line.strip()
        line_data = line.split()
        
        if nn == 0:
            app_header = line_data
            #print len(app_header)
        else:
            cfg = line_data[0].strip()

            line_data = line_data[1:]
            file_data[cfg] = {}

            for idx, app in enumerate(app_header):
                file_data[cfg][app] = float(line_data[idx])
                if file_data[cfg][app] == -1:
                    file_data[cfg][app] = 0

    f.close()
    return file_data


def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z


def select_apps_in_ds(ds, app_keys):
    for cfg in ds:
        tmp = ds[cfg].copy()
        for app in ds[cfg]:
            if not app in app_keys:
                del tmp[app]

        ds[cfg] = tmp


def select_bar(data1, cfg, apps, append_geomean=False):
    bar = []

    for app in apps:
        if cfg in data1:
            bar.append(data1[cfg][app])
        else:
            bar.append(0)

    if append_geomean:
        bar.append(geomean(bar))

    return np.array(bar)


def select_cfg_line(ds, cfg):
    return {k: ds[cfg][k] for k in ds[cfg]}


def normalize_to2(ds, normalized_line):
    for row in ds:
        for col in ds[row]:
            if normalized_line[col] == 0:
                print(normalized_line, '  contains  zero !!!!!')
                ds[row][col] = 1
            else:
                ds[row][col] /= normalized_line[col]


def normalize_to3(ds, normalized_line):
    for row in ds:
        for col in ds[row]:
            ds[row][col] = normalized_line[col] / ds[row][col]


def normalize_to(ds, cfg):
    normalize_to2(ds, select_cfg_line(ds, cfg))


def div1(ds, ds2):
    for row in ds:
        for col in ds[row]:
            if ds2[row][col] >= 1:
                print(row, col)
                ds[row][col] /= ds2[row][col]
            else:
                print(ds2[row][col], row, col)

def autolabel(rects):
    nn = 0
    for rect in rects:
        height = rect.get_height()
        ss = '%.2f'
        p = 1.05 * height
        if height >= 8:
            #print rect
            ss = '%d'
            p = 8.2
            nn += 1
            #print nn
            offset = + rect.get_width()
            #if nn > 1:
            #    break

            print(rect.get_x() + offset, p, ss % height)

            plt.text(rect.get_x() + offset, p, ss % height, ha='center', va='bottom', size=9)



def plot(data1, app_order, app_rmap, cfg_order, cfg_rmap, error_data=None, second_data_y=None, storebar1name='bars.csv', geo_mean=True):
    color=iter(cm.rainbow(np.linspace(0,1,10)))
    next(color)

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['axes.linewidth'] = 2

    #matplotlib.rcParams.update({'font.size': 24})

    hatches = ['', '//', '\\\\', 'xx', '...', '//', '*', '\\\\', '---', '\\\\', '//', '', '...', '////', '\\\\\\', 'xxxx', '....', '//', '*', '\\\\', '---', '\\\\', '//', '', '...'] 

    figwidth = len(cfg_order) * len(app_order) + len(cfg_order)
    f, ax = plt.subplots(1,1, figsize=(28,5))

    ylabel = [a for a in app_order]

    if geo_mean:
        ylabel.append('GeoMean')

    num_apps = len(ylabel)
    
    bars = []
    for cfg in cfg_order:
        original_cfg_name = cfg_rmap[cfg]
        tmp = select_bar(data1, original_cfg_name, [app_rmap[app] for app in app_order], geo_mean)
        #print(cfg, np.asarray(tmp))
        bars.append(tmp)

    f = open(storebar1name, 'w')
    f.write('config,')
    for app in app_order:
        f.write(app)
        f.write(',')

    if geo_mean:
        f.write('GeoMean')
        
    f.write('\n')
    for i, cfg in enumerate(cfg_order):
        f.write(cfg)
        for v in bars[i]:
            f.write(',')
            f.write('%.6f' % v)
        f.write('\n')
    f.close()


    barwidth = 1.5

    y_pos = []
    offset = 0
    for i in range(len(ylabel)):
        y_pos.append(offset)

        offset += barwidth * len(cfg_order) + barwidth
        if i == len(ylabel) - 2:
            offset += 1.5

    y_pos = np.array(y_pos)
    
    ax.yaxis.grid(linestyle='-.', linewidth='0.5', color='gray')

    #y_pos = np.array([i * (len(cfg_order) + barwidth) for i in range(len(ylabel))])
    #print(y_pos)
    
    #ax.set_ylim([0, 15])
    for i, (bar, cfg) in enumerate(zip(bars, cfg_order)):
        ax.bar(y_pos + i * barwidth, bar, align='center', alpha=1, label = cfg, width=barwidth, hatch=hatches[i], edgecolor='black', color=next(color))
        #print(bar)

    if error_data != None:
        for i, (bar, cfg) in enumerate(zip(bars, cfg_order)):
            if (cfg == 'AP' or cfg == 'AP_ideal'):
                continue

            original_cfg_name = cfg_rmap[cfg]
            yerr = select_bar(error_data, original_cfg_name, [app_rmap[app] for app in app_order], False)
            yerr = np.array(yerr)
            print('yerr = ', yerr, '\n', 'bar = ', bar)
            print(i, np.max(yerr))
            ax.errorbar(y_pos[:-1] + i * barwidth, bar[:-1], yerr=yerr, fmt='none', color='red', width=barwidth)

    ax.margins(x=0.01)
    

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

    # Put a legend to the right of the current axis
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=len(cfg_order), fancybox=True, shadow=True, fontsize = 16)
    leg.get_frame().set_edgecolor('black')
    
    ax.yaxis.set_tick_params(labelsize=24)

    plt.xticks(y_pos + len(cfg_order) * barwidth / 2, ylabel, rotation=45, fontsize=25)

    if second_data_y != None:    
        ax2 = ax.twinx()
        utilization_data = []
        for cfg in cfg_order:
            original_cfg_name = cfg_rmap[cfg]
            tmp = select_bar(second_data_y, original_cfg_name, [app_rmap[app] for app in app_order], True)
            #print(cfg, tmp)
            utilization_data.append(tmp)

        #cluster = {}
        bars = []
        for i, (bar, cfg) in enumerate(zip(utilization_data, cfg_order)):
            #ax2.plot(  cluster)
            ax2.scatter(y_pos + i * barwidth, bar, edgecolor='red', color='white', marker='D')
            bars.append( (y_pos + i * barwidth, bar))
        
        for cluster_id in range(len(app_order)):
            x_data = []    
            y_data = []
            for i in range(len(cfg_order)):
                x_data.append(bars[i][0][cluster_id])
                y_data.append(bars[i][1][cluster_id])

            print(x_data, y_data)
            ax2.plot(x_data, y_data, color='orange', linestyle='--')

        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        

        return plt, (ax, ax2)

    #ax.set_yscale('log')
    return plt, ax
    

