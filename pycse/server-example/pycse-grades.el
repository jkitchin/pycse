(require 'help-mode)
(defun close-buffer-return-string ()
  "close the current buffer and return its contents"
  (interactive)
  (let ((contents nil))
    (setq contents (buffer-string))
    (kill-buffer (current-buffer))
    (message contents)))

(defun my-annotate2 (arg)
  "Goto end of file and add annotation with link to point, and go back to point.

if you use a prefix arg, then you will get a new buffer in org-mode to construct the note. Otherwise you will enter the note in the minibuffer."
  (interactive)
  (let ((POINT (point))
	(line-number (line-number-at-pos))
	(note nil))
    (org-mark-ring-push) ; save position
    ;; create annotation section if it isn't here
    (if (not (search-forward "* ANNOTATION" nil t))
        (progn
          (goto-char (point-max))
          (newline)
          (insert "* ANNOTATION\n"))
      (save-restriction
	(org-narrow-to-subtree)
	(goto-char (point-max))
	(newline))

      ; no prefix command
      (if (equal current-prefix-arg nil)
	  (setq note (read-string "Note: " nil nil "note")) ; no prefix, get minibuffer
	(; need to open a new buffer, enter org-mode and provide a command to return the buffer-string and kill the buffer
	 (generate-new-buffer "*annotation*")
	 (switch-to-buffer "*annotation*")
	 (org-mode)
	 
	 )) ; else
      (insert "- " (format "[[elisp:(goto-char %i)][line %i]]: %s\n" POINT line-number note))))


      ;; finally go back to where we came from
      (org-mark-ring-goto))


(defun my-annotate (note)
  "Goto end of file and add annotation with link to point, and go back to point.

TODO with a prefix arg it should open a temporary buffer in org-mode, and then C-c C-c should insert the buffer contents into the comment.

TODO I should also figure out if I can integrate this with a git revision. the line numbers are fragile, and change if you modify the file above the ANNOTATIONS heading.
"
  (interactive "sNote: ")
  (let ((POINT (point))
	(line-number (line-number-at-pos)))
    (org-mark-ring-push) ; save position
    ;; create annotation section if it isn't here
    (if (not (search-forward "* ANNOTATION" nil t))
        (progn
          (goto-char (point-max))
          (newline)
          (insert "* ANNOTATION\n"))
      (save-restriction
	(org-narrow-to-subtree)
	(goto-char (point-max))
	(newline))
      (insert "- " (format "[[elisp:(goto-char %i)][line %i]]: %s\n" POINT line-number note))
      ;; finally go back
      (org-mark-ring-goto))))

(global-set-key "\C-ci" 'my-annotate)

(defun g (arg)
  "insert grade property on current heading.
TODO: add graded by property"
  (interactive "sGrade: ")
  (org-entry-put nil "GRADE" arg))

(defun send-email-feedback ()
  "Send current buffer to #+EMAIL:"
  (interactive)
  (let ((body-text (buffer-string))
	(subject (format "%s - feedback on %s" 
			 (org-entry-get (point) "COURSE" t)
			 (org-entry-get (point) "ASSIGNMENT" t)))
	(email (org-entry-get (point) "EMAIL" t)))
    (mail-other-window)
    (insert email)
    (mail-subject)
    (insert subject)
    (mail-text)
    (insert body-text)))
