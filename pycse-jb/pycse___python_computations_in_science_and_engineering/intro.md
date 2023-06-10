# Welcome to pycse - Python Computations in Science and Engineering

This content was created from a series of blog posts that began more than 10 years ago. I have collected these, and ordered them in ways that make some sense. Python has changed over these years, and some things may be out of date, or not in the current style. I have only lightly modified these in converting from org-mode to notebooks. The most notable changes are changing the org-files to use Jupyter-python instead of the plain Python blocks. I deleted some content that does not work anymore.


You can find an older [pdf](https://github.com/jkitchin/pycse/blob/master/pycse.pdf) with all the previous content build from the org-files. The org-mode content is at https://github.com/jkitchin/pycse/tree/master/pycse-chapters. It has been updated to use jupyter-emacs, then exported to these notebooks. These notebooks are more current than those files, and I have cleaned them up to work better with jupyter-book.

I don't have current plans to update these much further. I have instead started a new project at https://pointbreezepubs.gumroad.com/ where newer, more up to date content is available. This work here will remain free. There is still a lot to learn from reading this code, even as it continues to get dated. 

Some notable differences in style as of 2023:

1. f-strings are preferred over .format in all cases.
2. Many of the solvers, optimizers and integrators in scipy have been replaced by newer functions that are more flexible.


```{tableofcontents}
```
