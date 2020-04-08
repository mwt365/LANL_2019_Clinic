#  coding:utf-8

"""
  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: Produce a gallery of images after a run
  Created: 03/12/20
"""

import argparse
import os
import subprocess
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Produce slides from a run of PNSPipe',
        prog="slides",
    )

    parser.add_argument('-p', '--plot', help="Show plots",
                        action='store_false')
    parser.add_argument(
        '-i', '--image', help="Show images", action='store_false')
    parser.add_argument(
        '--rows', help="Number of rows per page", type=int, default=2)
    parser.add_argument(
        '--cols', help="Number of columns", type=int, default=3)
    parser.add_argument(
        '-l', '--landscape', action='store_true'
    )
    parser.add_argument('args', nargs=argparse.REMAINDER)

    args = parser.parse_args()

    with open('tmp.tex', 'w') as tex:
        print(
            r"""
            \documentclass[11pt%s]{article}
            \usepackage{multicol}
            \usepackage{graphicx}

            """ % (", landscape" if args.landscape else ""), file=tex)
        if args.landscape:
            print(r"\usepackage[width=10in,height=7.5in]{geometry}", file=tex)
        else:
            print(r"\usepackage[width=7.5in,height=10in]{geometry}", file=tex)
        print(r"\begin{document}", file=tex)
        print(r"\begin{multicols}{%d}" % args.cols, file=tex)
        for arg in args.args:
            # Prepare a list of bundled items
            items = []
            dirs = sorted([x for x in os.listdir(arg)])
            for d in dirs:
                the_dir = os.path.join(arg, d)
                if not os.path.isdir(the_dir):
                    continue
                item = []
                if args.image:
                    item.append(os.path.join(the_dir, 'spectrogram.png'))
                if args.plot:
                    item.append(os.path.join(the_dir, 'follower.pdf'))
                for i in item:
                    print(r"\includegraphics[width=\columnwidth]{%s}" % i, file=tex)

        print(r"\end{multicols}", file=tex)
        print(r"\end{document}", file=tex)

    subprocess.call(["pdflatex", "tmp"])


