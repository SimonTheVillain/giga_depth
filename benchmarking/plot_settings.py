import matplotlib as plt


def get_line_style(experiment_name):
    #algorithms used in synthetic tests
    algorithms = ["GigaDepth76c1280LCN", "GigaDepth78Uc1920",
                  "DepthInSpaceFTSF",
                  "ActiveStereoNet", "ActiveStereoNetFull",
                  "connecting_the_dots_stereo",
                  "HyperDepth"]
    #algorithms used in real tests:
    algorithms = ["GigaDepth76c1280LCN", "GigaDepth78Uc1920",
                  "DepthInSpaceFTSF",
                  "ActiveStereoNet",
                  "connecting_the_dots",
                  "HyperDepth", "HyperDepthXDomain",
                  "SGBM"]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    settings = {"GigaDepth76c1280LCN": (colors[0], "solid"),
                "GigaDepth": (colors[0], "solid"),
                "GigaDepth78Uc1920": (colors[1], "solid"),
                "DepthInSpaceFTSF": (colors[3], "solid"),
                "ActiveStereoNet": (colors[4], "solid"),
                "ActiveStereoNetFull": (colors[4], "dashed"),
                "connecting_the_dots_stereo": (colors[5], "solid"),
                "connecting_the_dots": (colors[5], "solid"),
                "HyperDepth": (colors[6], "solid"),
                "HyperDepthXDomain": (colors[6], "dashed"),
                "SGBM": (colors[7], "solid")
                }
    return settings[experiment_name]
#print(plt.rcParams['axes.prop_cycle'].by_key()['color'])