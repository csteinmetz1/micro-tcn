import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

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
    plt.plot(tcn100['N'], tcn100['rtf'], marker=next(marker), label='TCN-100-C')
    plt.plot(tcn300['N'], tcn300['rtf'], marker=next(marker), label='TCN-300-C')
    plt.plot(tcn324['N'], tcn324['rtf'], marker=next(marker), label='TCN-324-N')
    plt.plot(tcn1000['N'], tcn1000['rtf'], marker=next(marker), label='TCN-1000-C')
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

    marker = itertools.cycle(('x', '+', '.', '^', '*')) 
    color = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    mk = next(marker)
    c = next(color)
    plt.plot(tcn100_gpu['N'], tcn100_gpu['rtf'], c=c, linestyle='-', label='TCN-100-C')
    plt.plot(tcn100_cpu['N'], tcn100_cpu['rtf'], c=c, linestyle='--')

    mk = next(marker)
    c = next(color)
    plt.plot(tcn300_gpu['N'], tcn300_gpu['rtf'], c=c, linestyle='-', label='TCN-300-C')
    plt.plot(tcn300_cpu['N'], tcn300_cpu['rtf'], c=c, linestyle='--')

    mk = next(marker)
    c = next(color)
    plt.plot(tcn324_gpu['N'], tcn324_gpu['rtf'], c=c, linestyle='-', label='TCN-324-N')
    plt.plot(tcn324_cpu['N'], tcn324_cpu['rtf'], c=c, linestyle='--')

    #mk = next(marker)
    #c = next(color)
    #plt.plot(tcn1000_gpu['N'], tcn1000_gpu['rtf'], c=c, linestyle='-', label='TCN-1000-C')
    #plt.plot(tcn1000_cpu['N'], tcn1000_cpu['rtf'], c=c, linestyle='--')

    mk = next(marker)
    c = next(color)
    plt.plot(lstm32_gpu['N'], lstm32_gpu['rtf'], c=c, linestyle='-', label='LSTM-32')
    plt.plot(lstm32_cpu['N'], lstm32_cpu['rtf'], c=c, linestyle='--')

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

    plt.xticks(tcn100_gpu['N'], rotation=-45)

    plt.xlim([32,65540])
    plt.ylim([0.01,1400])
    plt.yticks([0.01, 0.1, 1.0, 10, 100, 1000],  [f"{n}" for n in [0.01, 0.1, 1.0, 10, 100, 1000]])

    plt.ylabel("Realtime factor")
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

    df_gpu = pd.read_csv("speed_gpu.csv", index_col=0)
    #df_cpu = pd.read_csv("speed_cpu_leopold.csv", index_col=0)
    df_cpu = pd.read_csv("speed_cpu_macbook.csv", index_col=0)


    runtime_plot(df_gpu, name="speed_gpu")
    runtime_plot(df_cpu, name="speed_cpu")
    joint_runtime_plot(df_gpu, df_cpu)

