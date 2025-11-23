# copy this file in ~/.jupyter/jupyter_nbconvert_config.py
c = get_config()

c.LatexExporter.latex_elements = {
    "preamble": r"""
\usepackage{xcolor}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage{xfrac}
"""
}
