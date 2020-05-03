

import pandas as pd
from Pipeline.figure_page import figure_page


def noise_stats(filename: str, results: list, **kwargs):
    """
    Write this
    """
    fields = "figname;beta;lam1;lam2;amp;mean;stdev;chisq;prob".split(';')
    data = {key: [] for key in fields}
    for n, seg in enumerate(results):
        noises = seg['noise']
        for k in data.keys():
            try:
                data[k].append(noises[k])
            except:
                if k == 'beta':
                    data[k].append(None)
                    data['lam1'].append(noises['lamb'])
                    data['lam2'].append(None)
                    data['stdev'].append(noises['mean'])
    figures = data.pop('figname')
    df = pd.DataFrame(data)
    print(df.to_string(sparsify=False))
    df.to_csv(f"{filename}-noise.csv")

    # Now produce a page showing all the figures
    figure_page(f"{filename}-figs", figures)
