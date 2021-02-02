import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def compute_receptive_field(kernel_size, nblocks, dilation_growth, stack_size=10):
    """ Compute the receptive field in samples."""
    rf = kernel_size
    layer_rf = []
    for n in range(1,nblocks):
        layer_rf.append(rf)
        dilation = dilation_growth ** (n % stack_size)
        rf = rf + ((kernel_size-1) * dilation)
    layer_rf.append(rf)
    return rf, layer_rf 

def plot_recepetive_field_growth():

    models = [
        {"name" : "TCN-100",
         "nblocks" : 4,
         "dilation_growth" : 10,
         "kernel_size" : 5,
         "color" : "#4053d3"
        }, 
        {"name" : "TCN-300",
         "nblocks" : 4,
         "dilation_growth" : 10,
         "kernel_size" : 13,
         "color" : "#ddb310"
        }, 
        {"name" : "TCN-324",
         "nblocks" : 10,
         "dilation_growth" : 2,
         "kernel_size" : 15,
         "color" : "#b51d14"
        }, 
        {"name" : "TCN-1000",
         "nblocks" : 5,
         "dilation_growth" : 10,
         "kernel_size" : 5,
         "color" : "#00b25d"
        }, 
    ]

    fig, ax = plt.subplots()
    width = 0.3
    sample_rate = 44100

    for idx, model in enumerate(models):
        rf, layer_rf = compute_receptive_field(model["kernel_size"], 
                                               model["nblocks"], 
                                               model["dilation_growth"])
        layers = np.arange(len(layer_rf)) + 1
        print(model["name"], layer_rf)
        plt.plot(layers, (np.array(layer_rf)/sample_rate) * 1e3, label=model["name"], marker='o', color=model["color"])
        plt.hlines((layer_rf[-1]/sample_rate) * 1e3, 1, 10, linestyle='--', colors=model["color"])
        #plt.text(layers[-1] + 0.2, layer_rf[-1] + 1250, model["name"])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    #plt.yscale('log')
    plt.xticks(np.arange(10) + 1)
    plt.xticks(np.arange(100, step=10) + 1)

    plt.legend()
    plt.grid(color='lightgray')

    fig.savefig("plots/receptive_field_growth.pdf")
    fig.savefig("plots/receptive_field_growth.svg")
    fig.savefig("plots/receptive_field_growth.png")


def runtime_plot(df, name="speed"):

    tcn100  = df[df['model_id'] == 'TCN-100-C']
    tcn300  = df[df['model_id'] == 'TCN-300-C']
    tcn324  = df[df['model_id'] == 'TCN-324-N']
    tcn370  = df[df['model_id'] == 'TCN-370-C']
    tcn1000 = df[df['model_id'] == 'TCN-1000-C']
    lstm32  = df[df['model_id'] == 'LSTM-32-C']

    fig, ax = plt.subplots(figsize=(5,3))

    marker = itertools.cycle(('x', '+', '.', '^', '*')) 

    #plt.plot(tcn370['N'], tcn370['rtf'], label='TCN-370-C')
    #plt.plot(tcn100['N'], tcn100['rtf'], marker=next(marker), label='TCN-100-C')
    plt.plot(tcn324['N'], tcn324['rtf'], marker=next(marker), label='TCN-324-N')
    plt.plot(tcn300['N'], tcn300['rtf'], marker=next(marker), label='TCN-300-C')

    #plt.plot(tcn1000['N'], tcn1000['rtf'], marker=next(marker), label='TCN-1000-C')
    plt.plot(lstm32['N'], lstm32['rtf'], marker=next(marker), label='LSTM-32')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.spines['left'].set_visible(False)

    ax.set_yscale('log', base=10)
    ax.set_xscale('log', base=2)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.xticks(tcn100['N'], rotation=-45)

    plt.xlim([32,65540])
    if name == "speed_gpu": 
        plt.ylim([0.2,1400])
    else:
        plt.yticks([0.01, 0.1, 1.0, 10], ['0.01', '0.1', '1.0', '10'])
        plt.ylim([0.01,20])

    plt.legend()
    plt.grid(c='lightgray')

    plt.hlines(1, 32, 65536, linestyles='dashed', color='k')
    plt.tight_layout()

    plt.savefig(f'plots/{name}.png')
    plt.savefig(f'plots/{name}.pdf')
    plt.savefig(f'plots/{name}.svg')
    plt.close('all')

def joint_runtime_plot(df_gpu, df_cpu):    
    
    tcn100_gpu  = df_gpu[df_gpu['model_id'] == 'TCN-100-C']
    tcn300_gpu  = df_gpu[df_gpu['model_id'] == 'TCN-300-C']
    tcn324_gpu  = df_gpu[df_gpu['model_id'] == 'TCN-324-N']
    tcn1000_gpu = df_gpu[df_gpu['model_id'] == 'TCN-1000-C']
    lstm32_gpu  = df_gpu[df_gpu['model_id'] == 'LSTM-32-C']

    tcn100_cpu  = df_cpu[df_cpu['model_id'] == 'TCN-100-C']
    tcn300_cpu  = df_cpu[df_cpu['model_id'] == 'TCN-300-C']
    tcn324_cpu  = df_cpu[df_cpu['model_id'] == 'TCN-324-N']
    tcn1000_cpu = df_cpu[df_cpu['model_id'] == 'TCN-1000-C']
    lstm32_cpu  = df_cpu[df_cpu['model_id'] == 'LSTM-32-C']

    fig, ax = plt.subplots(figsize=(6,3.5))

    N = 2048
    print("CPU")
    print("-" * 32)
    print("non-causal")
    tcn100_N_cpu  = df_cpu[df_cpu['model_id'] == 'TCN-100-N']
    tcn300_N_cpu  = df_cpu[df_cpu['model_id'] == 'TCN-300-N']
    tcn1000_N_cpu  = df_cpu[df_cpu['model_id'] == 'TCN-1000-N']

    print(tcn324_cpu[tcn324_cpu['N'] == N])
    print(tcn100_N_cpu[tcn100_N_cpu['N'] == N])
    print(tcn300_N_cpu[tcn300_N_cpu['N'] == N])
    print(tcn1000_N_cpu[tcn1000_N_cpu['N'] == N])

    print()
    print("causal")
    print(tcn100_cpu[tcn100_cpu['N'] == N])
    print(tcn300_cpu[tcn300_cpu['N'] == N])
    print(tcn1000_cpu[tcn1000_cpu['N'] == N])
    print(lstm32_cpu[lstm32_cpu['N'] == N])
    print()

    tcn324_16_cpu  = df_cpu[df_cpu['model_id'] == 'TCN-324-16-N']
    tcn324_8_cpu  = df_cpu[df_cpu['model_id'] == 'TCN-324-8-N']
    print(tcn324_16_cpu[tcn324_16_cpu['N'] == N])
    print(tcn324_8_cpu[tcn324_8_cpu['N'] == N])

    print("GPU")
    print("-" * 32)
    print("non-causal")
    tcn100_N_gpu  = df_gpu[df_gpu['model_id'] == 'TCN-100-N']
    tcn300_N_gpu  = df_gpu[df_gpu['model_id'] == 'TCN-300-N']
    tcn1000_N_gpu  = df_gpu[df_gpu['model_id'] == 'TCN-1000-N']

    print(tcn324_gpu[tcn324_gpu['N'] == N])
    print(tcn100_N_gpu[tcn100_N_gpu['N'] == N])
    print(tcn300_N_gpu[tcn300_N_gpu['N'] == N])
    print(tcn1000_N_gpu[tcn1000_N_gpu['N'] == N])

    print()
    print("causal")
    print(tcn100_gpu[tcn100_gpu['N'] == N])
    print(tcn300_gpu[tcn300_gpu['N'] == N])
    print(tcn1000_gpu[tcn1000_gpu['N'] == N])
    print(lstm32_gpu[lstm32_gpu['N'] == N])
    print()

    cn324_16_gpu  = df_gpu[df_gpu['model_id'] == 'TCN-324-16-N']
    cn324_8_gpu  = df_gpu[df_gpu['model_id'] == 'TCN-324-8-N']
    print(cn324_16_gpu[cn324_16_gpu['N'] == N])
    print(cn324_8_gpu[cn324_8_gpu['N'] == N])


    marker = itertools.cycle(('x', '+', '.', '^', '*')) 
    color = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    #mk = next(marker)
    #c = next(color)
    #plt.plot(tcn100_gpu['N'], tcn100_gpu['rtf'], c=c, linestyle='-', label='TCN-100-C')
    #plt.plot(tcn100_cpu['N'], tcn100_cpu['rtf'], c=c, linestyle='--')

    mk = next(marker)
    c = next(color)
    plt.plot(tcn324_gpu['N'], tcn324_gpu['rtf'], c=c, linestyle='-', label='TCN-324-N', marker=mk)
    plt.plot(tcn324_cpu['N'], tcn324_cpu['rtf'], c=c, linestyle='--', marker=mk)

    mk = next(marker)
    c = next(color)
    plt.plot(tcn300_gpu['N'], tcn300_gpu['rtf'], c=c, linestyle='-', label='TCN-300-C', marker=mk)
    plt.plot(tcn300_cpu['N'], tcn300_cpu['rtf'], c=c, linestyle='--', marker=mk)

    #mk = next(marker)
    #c = next(color)
    #plt.plot(tcn1000_gpu['N'], tcn1000_gpu['rtf'], c=c, linestyle='-', label='TCN-1000-C')
    #plt.plot(tcn1000_cpu['N'], tcn1000_cpu['rtf'], c=c, linestyle='--')

    mk = next(marker)
    c = next(color)
    plt.plot(lstm32_gpu['N'], lstm32_gpu['rtf'], c=c, linestyle='-', label='LSTM-32', marker=mk)
    plt.plot(lstm32_cpu['N'], lstm32_cpu['rtf'], c=c, linestyle='--', marker=mk)

    #plt.plot(tcn300['N'], tcn300['rtf'], marker=next(marker), label='TCN-300-C')
    #plt.plot(tcn324['N'], tcn324['rtf'], marker=next(marker), label='TCN-324-N')
    #plt.plot(tcn1000['N'], tcn1000['rtf'], marker=next(marker), label='TCN-1000-C')
    #plt.plot(lstm32['N'], lstm32['rtf'], marker=next(marker), label='LSTM-32')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.spines['left'].set_visible(False)

    ax.set_yscale('log', base=10)
    ax.set_xscale('log', base=2)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.xticks(tcn324_cpu['N'], rotation=-45)

    plt.xlim([32,65540])
    plt.ylim([0.01,1400])
    plt.yticks([0.01, 0.1, 1.0, 10, 100, 1000],  [f"{n}" for n in [0.01, 0.1, 1.0, 10, 100, 1000]])

    plt.ylabel("Real-time factor")
    plt.xlabel("Frame size")

    plt.legend()
    plt.grid(c='lightgray')

    plt.hlines(1, 32, 65536, linestyles='solid', color='k', linewidth=1)
    plt.tight_layout()

    plt.savefig(f'plots/speed_cpu+gpu.png')
    plt.savefig(f'plots/speed_cpu+gpu.pdf')
    plt.savefig(f'plots/speed_cpu+gpu.svg')
    plt.close('all')


if __name__ == '__main__':
    if not os.path.isdir("plots"):
        os.makedirs("plots")

    df_gpu = pd.read_csv("speed_gpu_rtx3090.csv", index_col=0)
    df_cpu = pd.read_csv("speed_cpu_macbook_v2.csv", index_col=0)

    #runtime_plot(df_gpu, name="speed_gpu")
    #runtime_plot(df_cpu, name="speed_cpu")
    joint_runtime_plot(df_gpu, df_cpu)

