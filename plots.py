import numpy as np
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


if __name__ == '__main__':
    plot_recepetive_field_growth()

        

