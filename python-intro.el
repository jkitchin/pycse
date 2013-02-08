(require 'browse-url)
(org-add-link-type "numpy"
;; FOLLOW code
  (lambda (keyword)
    (browse-url (format "http://docs.scipy.org/doc/numpy/reference/generated/numpy.%s.html" keyword)))
;; FORMAT code
  (lambda (keyword description format)
   (cond
    ((eq format 'html)
     (format "<a href=\"http://docs.scipy.org/doc/numpy/reference/generated/numpy.%s.html\">%s</a>" keyword keyword))
    ((eq format 'latex)
     (format "\\href{http://docs.scipy.org/doc/numpy/reference/generated/numpy.%s.html}{%s}"  keyword keyword)))))
