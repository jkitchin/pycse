The PYCSE blog
==================

These chapters are collected from my blog at [https://kitchingroup.cheme.cmu.edu](https://kitchingroup.cheme.cmu.edu) that started around 2013. The entries are roughly organized by category, and they were not written in the order presented. Many of them are translations of a [Matlab blog](http://matlab.cheme.cmu.edu/) to show that Python can be used for any problem I had previously used Matlab for. Today it is unambiguous, Python is 100% suitable for the vast majority of computations in science and engineering.

A lot has changed in Python over the last 10 years. These examples still work, and have been updated where needed. The style of many entries is the style I used 10 years ago, and today I would not solve these problems this way. I have not updated these entries for newer styles. An unfortunate consequence of converting these entries to Jupyter Book is the dates and chronology of the posts was not preserved. You can still go back to the [https://kitchingroup.cheme.cmu.edu](https://kitchingroup.cheme.cmu.edu) if you want to see the original source.

You can find an older [pdf](https://github.com/jkitchin/pycse/blob/master/pycse.pdf) with all the previous content build from the org-files. The org-mode content is at https://github.com/jkitchin/pycse/tree/master/pycse-chapters. It has been updated to use jupyter-emacs, then exported to these notebooks. These notebooks are more current than those files, and I have cleaned them up to work better with jupyter-book.

I don't have current plans to update these much further. I have instead started a new project at https://pointbreezepubs.gumroad.com/ where newer, more up to date content is available. This work here will remain free. There is still a lot to learn from reading this code, even as it continues to get dated. 

Some notable differences in style as of 2023:

1. f-strings are preferred over .format in all cases.
2. Many of the solvers, optimizers and integrators in scipy have been replaced by newer functions that are more flexible.
