import subprocess


def figure_page(fname, files, cols=3, landscape=True):
    """
    Produce a page of figures using latex
    """
    with open(fname + ".tex", 'w') as f:
        f.write(r"\documentclass[10pt")
        if landscape:
            f.write(',landscape]{article}\n')
            f.write(
                r'\usepackage[textheight=7.5in,textwidth=10in]{geometry}')
        else:
            f.write(']{article}\n')
            f.write(
                r'\usepackage[textheight=10in,textwidth=7.5in]{geometry}')
        print(r"""

        \usepackage{graphicx}
        \usepackage{longtable}
        \begin{document}
        """, file=f)
        f.write("\\begin{longtable}{" + "c" * cols + "}\n")
        width = round(0.9 / cols, 3)
        ig = "\\" + f"includegraphics[width={width}"
        ig += "\\textwidth]{"
        for n, graphic in enumerate(files):
            f.write(ig + graphic + "}\n")
            f.write("\\\\\n" if (n + 1) % cols == 0 else " & \n")
        print(r"\end{longtable}", file=f)
        print(r"\end{document}", file=f)
    subprocess.run(['pdflatex', fname])
