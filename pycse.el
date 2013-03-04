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


(require 'org-publish)
(setq org-publish-project-alist
      '(
	("pycse-content"
	 :base-directory "~/Dropbox/pycse/"
	 :base-extension "org"
	 :publishing-directory "~/Dropbox/pycse/gh-pages/"
	 :recursive t
	 :publishing-function org-publish-org-to-html
	 :headline-levels 4             ; Just the default for this project.
	 :auto-preamble t
	 )
	("pycse-static"
	 :base-directory "~/Dropbox/pycse/"
	 :base-extension "css\\|js\\|png\\|jpg\\|gif\\|pdf\\|mp3\\|ogg\\|swf\\|dat\\|mat\\|txt"
	 :publishing-directory "~/Dropbox/pycse/gh-pages/"
         :exclude "gh-pages"
	 :recursive t
	 :publishing-function org-publish-attachment
	 )
	;; ... all the components ...
	("pycse" :components ("pycse-content" "pycse-static"))
      ))

(org-publish "pycse")
