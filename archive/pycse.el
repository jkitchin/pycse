(require 'browse-url)
(org-add-link-type "numpy"
;; FOLLOW code
  (lambda (keyword)
    (browse-url (format "http://docs.scipy.org/doc/numpy/reference/generated/numpy.%s.html" keyword)))
;; FORMAT code
  (lambda (keyword description format)
   (cond
    ((eq format 'html)
     (format "<a href=\"http://docs.scipy.org/do
c/numpy/reference/generated/numpy.%s.html\">%s</a>" keyword keyword))
    ((eq format 'latex)
     (format "\\href{http://docs.scipy.org/doc/numpy/reference/generated/numpy.%s.html}{%s}"  keyword keyword)))))


(require 'ox-publish)
(setq org-publish-timestamp-directory "/home-research/jkitchin/Dropbox/books/pycse/"
 org-publish-project-alist
      '(
	("pycse-content"
	 :base-directory "~/Dropbox/books/pycse/"
	 :base-extension "org"
	 :publishing-directory "~/Dropbox/books/pycse/gh-pages/"
	 :publishing-function org-html-publish-to-html
	 :exclude "gh-pages\\|sandbox\\|pycse\\|octave"
	 :recursive t
	 :headline-levels 4             ; Just the default for this project.
	 :auto-preamble t)
	("pycse-static"
	 :base-directory "~/Dropbox/books/pycse/"
	 :base-extension "css\\|js\\|png\\|jpg\\|gif\\|pdf\\|mp3\\|ogg\\|swf\\|dat\\|mat\\|txt\\|svg"
	 :publishing-directory "~/Dropbox/books/pycse/gh-pages/"
	 :publishing-function org-publish-attachment
         :exclude "gh-pages\\|sandbox\\|octave\\|dist"
	 :recursive t)
	;; ... all the components ...
	("pycse" :components ("pycse-static" "pycse-content"))))

(org-publish "pycse" t)

