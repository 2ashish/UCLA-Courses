;;;;;;;;;;;;;;;;;;;;;;;
; Homework 4 ;;;;;;;;;;
; Ashish Kumar Singh ;;
; UID: 105479019 ;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;


; EXERCISE: Modify this function to decide satisifiability of delta.
; If delta is satisfiable, sat? returns a list of n integers that represents a model of delta,  
; otherwise it returns NIL. (See spec for details.)
; param n: number of variables in delta
; param delta: a CNF represented as a list of lists
(defun sat? (n delta)
    (backtrack n delta NIL)
)


(defun check-clause (clause nodes)
    (cond
        ((null clause)NIL)
        (t T)
    )
)

(defun check-cnf (delta nodes) 
    (cond
        ((null delta)T)
        ((null nodes)T)
        (t (and (check-clause (car delta) nodes) (check-cnf (cdr delta) nodes) ))
    )
)

(defun addnode2 (var nodes)
    (append nodes (list var))
)

(defun remove-clause (delta val)
    (cond
        ((null delta)NIL)
        (t (let*
                (
                    (clause (car delta))
                    (num (count val clause))
                    (clause-rest (cdr delta))
                )
                (if (> num 0) (remove-clause clause-rest val) (append (list clause) (remove-clause clause-rest val) ) )
            )
        )
    )
)


(defun remove-literal-row (clause literal) 
    (cond
        ((null clause)NIL)
        (t (let*    
                (
                    (l1 (car clause))
                    (rest-ans (remove-literal-row (cdr clause) literal))
                )
                (if (= literal l1) rest-ans (append (list l1) rest-ans ) )
            ) 
        )
    )
)

(defun remove-literal (delta literal) 
    (cond
        ((null delta)NIL)
        (t (append (list (remove-literal-row (car delta) literal)) (remove-literal (cdr delta) literal) ))
    )
)

(defun flatten-append (E1 E2)
    ;; Inputs: E1 and E2 are two lisp expression with atoms as numbers
    ;; Output: A list with all atoms of E2 appended to E1
    (cond
        ((null E2) E1 )
        ((atom E2) (append E1 (cons E2 NIL) ))
        (t 
            (flatten-append (flatten-append E1 (car E2)) (cdr E2))
        )
    )
)
;function to decide variable ordering
(defun next-var (delta cur max-var max-val nodes) ;variable ordering
    (cond
        ((= cur 0)max-var)
        ((> (+ (count cur nodes) (count (- 0 cur) nodes) ) 0) (next-var delta (- cur 1) max-var max-val nodes))
        (t (let*
                (
                    (pc (count cur delta))
                    (nc (count (- 0 cur) delta))
                    (c (+ pc nc))
                )
                (if (> c max-val) (next-var delta (- cur 1) cur c nodes) (next-var delta (- cur 1) max-var max-val nodes) )
            )
        )
    )
)

;find single literal clause
(defun single-literal (delta)
    (cond
        ((null delta)NIL)
        ((= (length (car delta)) 1)(caar delta))
        (t (single-literal (cdr delta)))
    )
)

;find single literal clause or literal used maximum times
(defun next-var-base (delta cur max-var max-val nodes)
    (let
        (
            (check (single-literal delta))
            (delta-flatten (flatten-append '() delta))
        )
        (if (null check) (next-var delta-flatten cur max-var max-val nodes) check )
    )
)

;backtrack over all possible configurations
(defun backtrack (n delta nodes) 
    (let
        ((len (length nodes)))
        (cond
            ((null (check-cnf delta nodes))NIL)
            ((= len n) nodes)
            (t (let*
                    (
                        (delta-flatten (flatten-append '() delta))
                        (var (next-var-base delta n (+ n 1) -1 nodes))
                        (nvar (- 0 var))
                        (pc (count var delta-flatten))
                        (nc (count nvar delta-flatten))
                    )
                    (if (> pc nc) ;value ordering
                        (or (let*
                                (
                                    (nodes-updated (addnode2 var nodes))
                                    (delta-updated (remove-literal (remove-clause delta var ) nvar ))
                                )
                                (backtrack n delta-updated nodes-updated)
                            )
                            (let*
                                (
                                    (nodes-updated (addnode2 nvar nodes))
                                    (delta-updated (remove-literal (remove-clause delta nvar ) var ))
                                )
                                (backtrack n delta-updated nodes-updated)
                            )
                            
                        )
                        (or (let*
                                (
                                    (nodes-updated (addnode2 nvar nodes))
                                    (delta-updated (remove-literal (remove-clause delta nvar ) var ))
                                )
                                (backtrack n delta-updated nodes-updated)
                            )
                        (let*
                                (
                                    (nodes-updated (addnode2 var nodes))
                                    (delta-updated (remove-literal (remove-clause delta var ) nvar ))
                                )
                                (backtrack n delta-updated nodes-updated)
                            )
                        )
                    )
                )
            )
        )
    )
)



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Functions that help you parse CNF from files in folder cnfs/
; You need not modify any functions in this section
; Usage (solve-cnf <path-to-file>)
; e.g., (solve-cnf "./cnfs/f1/sat_f1.cnf")
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(defun split-line (line)
  (if (equal line :eof)
      :eof
      (with-input-from-string (s line) (loop for x = (read s nil) while x collect x))))

(defun read-cnf (filename)
  (with-open-file (in filename)
    (loop for line = (split-line (read-line in nil :eof)) until (equal line :eof)
      if (equal 'p (first line)) collect (third line)      ; var count
      if (integerp (first line)) collect (butlast line)))) ; clause

(defun parse-cnf (filename)
  (let ((cnf (read-cnf filename))) (list (car cnf) (cdr cnf))))

; Following is a helper function that combines parse-cnf and sat?
(defun solve-cnf (filename)
  (let ((cnf (parse-cnf filename))) (sat? (first cnf) (second cnf))))


;testcases
;(write (sat? 3 '((1 -2 3)(-1)(-2 -3)) ))                (FRESH-LINE)
;(write (sat? 1 '((1) (-1))))                            (FRESH-LINE)
;(write (flatten-append '() '((1 -2 3)(-1)(-2 -3)) ))    (FRESH-LINE)
;(write (remove-literal '((1) (1 2 3)) 1 ))                (FRESH-LINE)


