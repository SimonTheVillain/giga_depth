import pickle
import matplotlib.pyplot as plt

data = pickle.load( open( "shapenet_results.pkl", "rb" ) )

#todo: create colormaps for specific frame!!!!

subfigures = True

if subfigures:
    fig, axs = plt.subplots(1, 2)
    x = data["absolute_outlier_ths"]
    for key, val in data["results"].items():
        axs[0].plot(x[10:], val["outlier_ratios"][10:], label=key)


    #axs[0].xticks([0.5, 1, 2, 3, 4, 5])
    # show a legend on the plot
    axs[0].legend()

    x = data["relative_outlier_ths"]
    th = data["relative_th"]
    for key, val in data["results"].items():
        axs[1].plot(x, val["relative_outlier_ratios"], label=key)

    # show a legend on the plot
    axs[1].legend()
else:
    print(data)
    x = data["absolute_outlier_ths"]


    plt.figure(1)
    for key, val in data["results"].items():
        plt.plot(x[10:], val["outlier_ratios"][10:], label=key)


    plt.xticks([0.5, 1, 2, 3, 4, 5])
    # show a legend on the plot
    plt.legend()

    plt.figure(2)

    x = data["relative_outlier_ths"]
    th = data["relative_th"]
    for key, val in data["results"].items():
        plt.plot(x, val["relative_outlier_ratios"], label=key)

    # show a legend on the plot
    plt.legend()

# Display the figures.
plt.show(block=False)
plt.waitforbuttonpress()
plt.pause(5.)