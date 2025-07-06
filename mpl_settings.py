import matplotlib as mpl
import matplotlib.pyplot as plt
from consts import Path
import matplotlib.font_manager


# print([f.name for f in matplotlib.font_manager.fontManager.ttflist])

def fmt(x, val):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    # return r'${} \times 10^{{{}}}$'.format(a, b)
    return rf'{a}E+{b:02}'


def mpl_style_params(new_rcparams=None):
    rcParams = mpl.rcParams
    # rcParams['lines.linestyle'] = '--'
    # rcParams['legend.fontsize'] = 'large' #'x-large'
    rcParams['legend.shadow'] = False
    rcParams['lines.marker'] = 'o'
    rcParams['lines.markersize'] = 2
    # rcParams['lines.linewidth'] = 3.5  # 2
    rcParams['ytick.major.width'] = 2.5
    rcParams['xtick.major.width'] = 2.5
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['axes.grid'] = True
    rcParams['figure.autolayout'] = False
    rcParams['savefig.format'] = 'png'
    rcParams["scatter.marker"] = "o"  # "x"
    rcParams.update({'font.size': 16})

    # Say, "the default sans-serif font is COMIC SANS"
    # rcParams['font.sans-serif'] = 'Liberation Sans'
    # Then, "ALWAYS use sans-serif fonts"
    # rcParams['font.family'] = "sans-serif"
    rcParams.update({
        "text.usetex": True,  # Use LaTeX to write all text
        # "pgf.texsystem": "pdflatex",
        # Is my latex installation broken???
        #"font.family": "serif", # "serif",  # Use serif fonts
        #"font.serif": ["Computer Modern"],  # Ensure it matches LaTeX default font

        # "text.latex.preamble": r"\\usepackage{amsmath}"  # Add more packages as needed
        #"pgf.preamble": "\n".join([ # plots will use this preamble
        #r"\usepackage[utf8]{inputenc}",
        #r"\usepackage[T1]{fontenc}",
        #r"\usepackage{siunitx}",
        #])
    })
    # rcParams["savefig.directory"] = ""

    if new_rcparams:
        rcParams.update(new_rcparams)

    Path(rcParams["savefig.directory"]).mkdir(parents=True, exist_ok=True)

    return rcParams


if __name__ == '__main__':
    rcParams = mpl_style_params()
    mpl.rcParams.update(rcParams)

    from matplotlib.pyplot import subplots, xlabel, ylabel, grid, show
    s = fr"Mean (-): (74.06$\pm$2.56) $\mu$m\\Nominal (--): 73 $\mu$m"

    fig, ay = subplots()
    plt.text(0.5, 0.5, s)
    # Using the specialized math font elsewhere, plus a different font
    xlabel(r"The quick brown fox jumps over the lazy dog", fontsize=18)
    # No math formatting, for comparison
    ylabel(r'Italic and just Arial and not-math-font', fontsize=18)
    grid()
    # plt.savefig("abe.png")
    show()
