; Updated by Ashish Kumar Singh
; Date: April 20, 2022
; CS161 Hw3: Sokoban
; 
; *********************
;    READ THIS FIRST
; ********************* 
;
; All functions that you need to modify are marked with 'EXERCISE' in their header comments.
; Do not modify a-star.lsp.
; This file also contains many helper functions. You may call any of them in your functions.
;
; *Warning*: The provided A* code only supports the maximum cost of 4999 for any node.
; That is f(n)=g(n)+h(n) < 5000. So, be careful when you write your heuristic functions.
; Do not make them return anything too large.
;
; For Allegro Common Lisp users: The free version of Allegro puts a limit on memory.
; So, it may crash on some hard sokoban problems and there is no easy fix (unless you buy 
; Allegro). 
; Of course, other versions of Lisp may also crash if the problem is too hard, but the amount
; of memory available will be relatively more relaxed.
; Improving the quality of the heuristic will mitigate this problem, as it will allow A* to
; solve hard problems with fewer node expansions.
; 
; In either case, this limitation should not significantly affect your grade.
; 
; Remember that most functions are not graded on efficiency (only correctness).
; Efficiency can only influence your heuristic performance in the competition (which will
; affect your score).
;  
;


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; General utility functions
; They are not necessary for this homework.
; Use/modify them for your own convenience.
;

;
; For reloading modified code.
; I found this easier than typing (load "filename") every time. 
;
(defun reload()
  (load "hw3.lsp")
  )

;
; For loading a-star.lsp.
;
(defun load-a-star()
  (load "a-star.lsp"))

;
; Reloads hw3.lsp and a-star.lsp
;
(defun reload-all()
  (reload)
  (load-a-star)
  )

;
; A shortcut function.
; goal-test and next-states stay the same throughout the assignment.
; So, you can just call (sokoban <init-state> #'<heuristic-name>).
; 
;
(defun sokoban (s h)
  (a* s #'goal-test #'next-states h)
  )

; shortcut function to run all games at once, used for timing
(defun sokoban-all (h)
  (list 
    (a* p1 #'goal-test #'next-states h)
    (a* p2 #'goal-test #'next-states h)
    (a* p3 #'goal-test #'next-states h)
    (a* p4 #'goal-test #'next-states h)
    (a* p5 #'goal-test #'next-states h)
    (a* p6 #'goal-test #'next-states h)
    (a* p7 #'goal-test #'next-states h)
    (a* p8 #'goal-test #'next-states h)
    (a* p9 #'goal-test #'next-states h)
    (a* p10 #'goal-test #'next-states h)
    (a* p11 #'goal-test #'next-states h)
    (a* p12 #'goal-test #'next-states h)
    ;(a* p13 #'goal-test #'next-states h)
    (a* p14 #'goal-test #'next-states h)
    ;(a* p15 #'goal-test #'next-states h)
  )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; end general utility functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; We now begin actual Sokoban code
;

; Define some global variables
(setq blank 0)
(setq wall 1)
(setq box 2)
(setq keeper 3)
(setq star 4)
(setq boxstar 5)
(setq keeperstar 6)

; Some helper functions for checking the content of a square
(defun isBlank (v)
  (= v blank)
  )

(defun isWall (v)
  (= v wall)
  )

(defun isBox (v)
  (= v box)
  )

(defun isKeeper (v)
  (= v keeper)
  )

(defun isStar (v)
  (= v star)
  )

(defun isBoxStar (v)
  (= v boxstar)
  )

(defun isKeeperStar (v)
  (= v keeperstar)
  )

;
; Helper function of getKeeperPosition
;
(defun getKeeperColumn (r col)
  (cond ((null r) nil)
	(t (if (or (isKeeper (car r)) (isKeeperStar (car r)))
	       col
	     (getKeeperColumn (cdr r) (+ col 1))
	     );end if
	   );end t
	);end cond
  )

;
; getKeeperPosition (s firstRow)
; Returns a list indicating the position of the keeper (c r).
; 
; Assumes that the keeper is in row >= firstRow.
; The top row is the zeroth row.
; The first (left) column is the zeroth column.
;
(defun getKeeperPosition (s row)
  (cond ((null s) nil)
	(t (let ((x (getKeeperColumn (car s) 0)))
	     (if x
		 ;keeper is in this row
		 (list x row)
		 ;otherwise move on
		 (getKeeperPosition (cdr s) (+ row 1))
		 );end if
	       );end let
	 );end t
	);end cond
  );end defun

;
; cleanUpList (l)
; returns l with any NIL element removed.
; For example, if l is '(1 2 NIL 3 NIL), returns '(1 2 3).
;
(defun cleanUpList (L)
  (cond ((null L) nil)
	(t (let ((cur (car L))
		 (res (cleanUpList (cdr L)))
		 )
	     (if cur 
		 (cons cur res)
		  res
		 )
	     );end let
	   );end t
	);end cond
  );end 


; Helper function for goal-test which check if there is a box or keeper 
; in the given row
(defun box-keeper-test-row (r)
    (let
        (
            (count-box (count box r))
            (count-keeper (count keeper r))
        )
        (if (= (+ count-box count-keeper) 0) NIL t)
    )
)

; This function returns true (t)
; if and only if s is a goal state of the game.
; (neither any boxes nor the keeper is on a non-goal square)
; If there is a box or keeper in any row we return NIL
(defun goal-test (s)
    (cond
        ((null s)t)
        (t  (let
                (
                    (row-check (box-keeper-test-row (car s)))
                )
                (if row-check NIL (goal-test (cdr s)))
            )
        )
    )
    
)


; get value in column row of the state
(defun get-square (s c r)
    (cond
        ((< c 0)wall)
        ((< r 0)wall)
        ((> r (- (length s) 1))wall)
        ((> c (- (length (car s)) 1))wall)
        (t (car (nthcdr c (car (nthcdr r s)))) )
    )
)

; set the coulmn in given row to value
(defun set-row-square (s c v)
    (cond
        ((> c 0)(append (list (car s)) (set-row-square (cdr s) (- c 1) v) ))
        ((= c 0)(append (list v ) (cdr s) ))
        (t s)
    )
)

;set the the required column row in state to the given valve
(defun set-square (s c r v)
    (cond
        ((> r 0)(append (list (car s)) (set-square (cdr s) c (- r 1) v) ))
        ((= r 0)(append (list (set-row-square (car s) c v)) (cdr s)))
        (t s)
    )
)

;input: position list of two element column and row and direction
; outputs: adjacent position (column,row) in the given direction
(defun getDirPos (pos dir)
    (let
        (
            (c (car pos))
            (r (cadr pos))
        )
        (cond
            ((= dir 1)(list c (- r 1) ))
            ((= dir 2)(list c (+ r 1) ))
            ((= dir 3)(list (- c 1) r ))
            ((= dir 4)(list (+ c 1) r ))
        )
    )
)

;Helper function for next-states, this function tries to move keeper in the dir 
; direction, if it is not possible to move it returns NIl, otherwise updated state
(defun try-move (s dir)
    (let*
        (
            (pos (getKeeperPosition s 0))
            (pos1 (getDirPos pos dir))
            (pos2 (getDirPos pos1 dir))
            (r (cadr pos))
            (r1 (cadr pos1))
            (r2 (cadr pos2))
            (c (car pos))
            (c1 (car pos1))
            (c2 (car pos2))
            (pos-val (get-square s c r))
            (pos-update (if (= pos-val keeper) blank star))
            (v1 (get-square s c1 r1))
            (v2 (get-square s c2 r2))
            (pos1-update (if (or (= v1 blank) (= v1 box)) keeper keeperstar))
        )
        (cond
            ((= v1 blank)(set-square (set-square s c r pos-update) c1 r1 pos1-update))
            ((= v1 wall)NIL)
            ((or (= v1 box) (= v1 boxstar))(cond
                            ((= v2 blank) (set-square (set-square (set-square s c r pos-update) c1 r1 pos1-update) c2 r2 box) )
                            ((or (= v2 wall) (= v2 box) (= v2 boxstar))NIL)
                            ((= v2 star) (set-square (set-square (set-square s c r pos-update) c1 r1 pos1-update) c2 r2 boxstar) )
                        )
            )
            ((= v1 star)(set-square (set-square s c r pos-update) c1 r1 keeperstar))
        )
    )    
)

; This function returns the list of sucessor states of s.
; We do this by tring all four possible moves and append in the list if it results in valid state
(defun next-states (s)
  (let* ((pos (getKeeperPosition s 0))
	 (x (car pos))
	 (y (cadr pos))
	 (UP 1)
	 (DOWN 2)
	 (LEFT 3)
	 (RIGHT 4)
	 ;x and y are now the coordinate of the keeper in s.
	 (result (list (try-move s UP) (try-move s RIGHT) (try-move s DOWN) (try-move s LEFT)))
	 ;(result (list (try-move s LEFT)))
	 )
    (cleanUpList result);end
   );end let
  );

;trivial heuristic return only 0
(defun h0 (s)
    0
)

; heuristic that return number of misplaced boxes
(defun h1 (s)
    (cond
        ((null s)0)
        (t (+ (count BOX (car s)) (h1 (cdr s))))
    )  
)


; this fuction return 1 if the given box is not adjacent to a goal, otherwise 2
(defun neargoal (box s)
    (let*
        (
            (c (car box))
            (r (cadr box))
            (v1 (get-square s (+ c 1) r))
            (v2 (get-square s (- c 1) r))
            (v3 (get-square s c (+ r 1)))
            (v4 (get-square s c (- r 1)))
            (check (or (= v1 star) (= v2 star) (= v3 star) (= v4 star)))
        )
        (if check 1 2)
    )
)

; this function returns number of misplaced boxes weighted by function neargoal
(defun neighbour (boxes s)
    (cond
        ((null boxes)0)
        (t (+ (neargoal (car boxes) s) (neighbour (cdr boxes) s) ))
    )
)

;this function check if the given box is in deadlock state, 
; i.e. surrounded by two adjacent walls
; this helps in speeding up the search as we dont need to branch out unsolvable states
(defun deadlock-cond (s box)
    (let*
        (
	        (c (car box))
	        (r (cadr box))
            (v1 (get-square s (- c 1) r)) ;up
            (v2 (get-square s c (+ r 1))) ;right
            (v3 (get-square s (+ c 1) r)) ;down
            (v4 (get-square s c (- r 1))) ;left
        )
        (or (and (= v1 wall) (= v2 wall) ) (and (= v2 wall) (= v3 wall)) (and (= v3 wall) (= v4 wall)) (and (= v4 wall) (= v1 wall)))
    )
    
)

; this function checks if there is any box in boxes which have deadlock, 
; it recursively calls deadlock-cond function which does the check
(defun deadlock (s boxes)
    (cond
        ((null boxes)NIL)
        (t (or (deadlock-cond s (car boxes)) (deadlock s (cdr boxes))))
    )
)

; heuristic that checks if there is a deadlock (unsolvable) box in the state, 
; if yes then it returns a high value 1000, otherwise it returns number of misplaced 
; boxes weighted with whether there is an adjacent goal
; This heuristic is admissible as the boxes which dont have any adjacent goals 
; needs to be moved by atleast 2
(defun h2 (s)
    (let*
        (
            (boxes (list-boxes-pos s 0))
            (check (deadlock s boxes))
        )
        (if check 1000 (neighbour boxes s))
    ) 
)




;;other attemps for heuristics which are not performing as well in terms of execution speed


;(defun getBoxesColumn (r col row)
;  (cond ((null r) nil)
;	(t (if (or (isBox (car r)))
;	        (append (list(list col row)) (getBoxesColumn (cdr r) (+ col 1) row))
;	     (getBoxesColumn (cdr r) (+ col 1) row)
;	     );end if
;	   );end t
;	);end cond
;  )
;
;(defun list-boxes-pos (s row)
;    (cond 
;        ((null s) nil)
;	    (t (let* 
;	            (
;	                (x (getBoxesColumn (car s) 0 row))
;	                (y (cleanUpList x))
;	           )
;	            (if y
;		            ;keeper is in this row
;		            (append y (list-boxes-pos (cdr s) (+ row 1)))
;		            ;otherwise move on
;		            (list-boxes-pos (cdr s) (+ row 1))
;		        );end if
;	       );end let
;	    );end t
;	);end cond
;)
;
;(defun getGoalsColumn (r col row)
;  (cond ((null r) nil)
;	(t (if (or (isStar (car r)) (isKeeperStar (car r)))
;	        (append (list(list col row)) (getGoalsColumn (cdr r) (+ col 1) row))
;	     (getGoalsColumn (cdr r) (+ col 1) row)
;	     );end if
;	   );end t
;	);end cond
;  )
;
;(defun list-goals-pos (s row)
;    (cond 
;        ((null s) nil)
;	    (t (let* 
;	            (
;	                (x (getGoalsColumn (car s) 0 row))
;	                (y (cleanUpList x))
;	           )
;	            (if y
;		            ;keeper is in this row
;		            (append y (list-goals-pos (cdr s) (+ row 1)))
;		            ;otherwise move on
;		            (list-goals-pos (cdr s) (+ row 1))
;		        );end if
;	       );end let
;	    );end t
;	);end cond
;)
;
;(defun dist (box goal)
;    (let
;        (
;            (b1 (car box))
;            (b2 (cadr box))
;            (g1 (car goal))
;            (g2 (cadr goal))
;        )
;        (+ (abs (- b1 g1) ) (abs (- b2 g2)))
;    )
;)
;
;(defun mindist (box goals)
;    (cond
;        ((null goals)0)
;        (t (min (dist box (car goals)) (mindist box (cdr goals))))
;    )
;)
;
;(defun getmindists (boxes goals)
;    (cond
;        ((null boxes)0)
;        (t (+ (mindist (car boxes) goals) (getmindists (cdr boxes) goals)))
;    )
;)

;(defun h21 (s)
;    (let
;        (
;            (boxes (list-boxes-pos s 0))
;            (goals (list-goals-pos s 0))
;        )
;        (getmindists boxes goals)
;    )
;
;)
;
;(defun h6 (s)
;    (let
;        (
;            (boxes (list-boxes-pos s 0))
;            (goals (list-goals-pos s 0))
;            (pos (getKeeperPosition s 0 ))
;        )
;        (+ (getmindists boxes goals) (getmindists (list pos) boxes) )
;    )
;
;)
;
;(defun h3 (s)
;    (let
;        (
;            (h (h1 s))
;        )
;        (if (> h 4) (h2 s) h)
;    )  
;)
;different implementation which makes the search faster but cannot prove admissibility
;(defun neargoal (box s)
;    (let*
;        (
;            (c (car box))
;            (r (cadr box))
;            (v1 (get-square s (+ c 1) r))
;            (v2 (get-square s (- c 1) r))
;            (v3 (get-square s c (+ r 1)))
;            (v4 (get-square s c (- r 1)))
;            (check (or (= v1 4) (= v2 4) (= v3 4) (= v4 4)))
;            (b (count 2 (list v1 v2 v3 v4)))
;        )
;        (if check (+ b 1) (+ b 2))
;    )
;)
;
;(defun neargoal (box s)
;    (let*
;        (
;            (c (car box))
;            (r (cadr box))
;            (v1 (get-square s (+ c 1) r))
;            (v2 (get-square s (- c 1) r))
;            (v3 (get-square s c (+ r 1)))
;            (v4 (get-square s c (- r 1)))
;            (check (or (= v1 4) (= v2 4) (= v3 4) (= v4 4)))
;        )
;        (if check 1 2)
;    )
;)
;
;(defun neighbour (boxes s)
;    (cond
;        ((null boxes)0)
;        (t (+ (neargoal (car boxes) s) (neighbour (cdr boxes) s) ))
;    )
;)
;
;(defun h4 (s)
;    (let
;        (
;            (boxes (list-boxes-pos s 0))
;        )
;        (neighbour boxes s)
;    ) 
;)

;(defun deadlock-cond (s box)
;    (let*
;        (
;	        (c (car box))
;	        (r (cadr box))
;            (v1 (get-square s (- c 1) r)) ;up
;            (v2 (get-square s c (+ r 1))) ;right
;            (v3 (get-square s (+ c 1) r)) ;down
;            (v4 (get-square s c (- r 1))) ;left
;        )
;        (or (and (= v1 1) (= v2 1) ) (and (= v2 1) (= v3 1)) (and (= v3 1) (= v4 1)) (and (= v4 1) (= v1 1)))
;    )
;    
;)
;
;(defun deadlock (s boxes)
;    (cond
;        ((null boxes)NIL)
;        (t (or (deadlock-cond s (car boxes)) (deadlock s (cdr boxes))))
;    )
;)
;
;(defun h45 (s)
;    (let*
;        (
;            (boxes (list-boxes-pos s 0))
;            (check (deadlock s boxes))
;        )
;        (if check 1000 (neighbour boxes s))
;    ) 
;)

;(defun h46 (s)
;    (let*
;        (
;            (boxes (list-boxes-pos s 0))
;            (check (deadlock s boxes))
;            (pos (getKeeperPosition s 0 ))
;        )
;        (if check 1000 (+ (neighbour boxes s) (getmindists (list pos) boxes) ))
;    ) 
;)
;
;(defun h47 (s)
;    (let*
;        (
;            (boxes (list-boxes-pos s 0))
;            (check (deadlock s boxes))
;            (pos (getKeeperPosition s 0 ))
;            (v (get-square s (car pos) (cadr pos)))
;        )
;        (if check 1000 (+ (neighbour boxes s) (if (= v 3) 1 0) ))
;    ) 
;)
;
;(defun h15 (s)
;    (let*
;        (
;            (boxes (list-boxes-pos s 0))
;            (check (deadlock s boxes))
;            (h (h1 s))
;        )
;        (if check 1000 h)
;    ) 
;)
;
;(defun h25 (s)
;    (let*
;        (
;            (boxes (list-boxes-pos s 0))
;            (check (deadlock s boxes))
;            (goals (list-goals-pos s 0))
;        )
;        (if check 1000 (getmindists boxes goals))
;    ) 
;)
;
;(defun h5 (s)
;    (let
;        (
;            (boxes (list-boxes-pos s 0))
;            (pos (getKeeperPosition s 0 ))
;        )
;        (+ (getmindists (list pos) boxes) (- (length boxes) 1))
;        ;(getmindists (list pos) boxes)
;    ) 
;)
;
;(defun h7 (s)
;    (let
;        (
;            (boxes (list-boxes-pos s 0))
;            (pos (getKeeperPosition s 0 ))
;        )
;        (+ (getmindists (list pos) boxes) (neighbour boxes s))
;        ;(getmindists (list pos) boxes)
;    ) 
;)



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

#|
 | Some predefined problems.
 | Each problem can be visualized by calling (printstate <problem>). For example, (printstate p1).
 | Problems are ordered roughly by their difficulties.
 | For most problems, we also provide a number which indicates the depth of the optimal solution.
 | These numbers are located at the comments of the problems. For example, the first problem below has optimal solution depth 6.
 | As for the solution depth, any admissible heuristic must make A* return an optimal solution. So, the depths of the optimal solutions provided could be used for checking whether your heuristic is admissible.
 |
 | Warning: some problems toward the end are quite hard and could be impossible to solve without a good heuristic!
 | 
 |#
;(6)

(setq p1 '((1 1 1 1 1 1)
	   (1 0 3 0 0 1)
	   (1 0 2 0 0 1)
	   (1 1 0 1 1 1)
	   (1 0 0 0 0 1)
	   (1 0 4 0 4 1)
	   (1 1 1 1 1 1)))

;(15)

(setq p2 '((1 1 1 1 1 1 1)
	   (1 0 0 0 0 0 1) 
	   (1 0 0 0 0 0 1) 
	   (1 0 0 2 1 4 1) 
	   (1 3 4 0 1 0 1)
	   (1 1 1 1 1 1 1)))

;(13)
(setq p3 '((1 1 1 1 1 1 1 1 1)
	   (1 0 0 0 1 0 0 0 1)
	   (1 0 0 0 2 0 3 4 1)
	   (1 0 0 0 1 0 0 0 1)
	   (1 0 4 0 1 0 0 0 1)
	   (1 1 1 1 1 1 1 1 1)))

;(17)
(setq p4 '((1 1 1 1 1 1 1)
	   (0 0 0 0 0 1 4)
	   (0 0 0 0 0 0 0)
	   (0 0 1 1 1 0 0)
	   (0 0 1 0 0 0 0)
	   (0 2 1 0 0 4 0)
	   (0 3 1 0 0 0 0)))

;(12)
(setq p5 '((1 1 1 1 1 1)
	   (1 1 0 0 1 1)
	   (1 0 0 0 0 1)
	   (1 4 2 2 4 1)
	   (1 0 0 0 4 1)
	   (1 1 3 1 1 1)
	   (1 1 1 1 1 1)))

;(13)
(setq p6 '((1 1 1 1 1 1 1 1)
	   (1 0 0 0 0 0 4 1)
	   (1 4 0 0 2 2 3 1)
	   (1 0 0 1 0 0 4 1)
	   (1 1 1 1 1 1 1 1)))

;(47)
(setq p7 '((1 1 1 1 1 1 1 1 1 1)
	   (0 0 1 1 1 1 4 0 0 3)
	   (0 0 0 0 0 1 0 0 0 0)
	   (0 0 0 0 0 1 0 0 1 0)
	   (0 0 1 0 0 1 0 0 1 0)
	   (0 2 1 0 0 0 0 0 1 0)
	   (0 0 1 0 0 0 0 0 1 4)))

;(22)
(setq p8 '((1 1 1 1 1 1)
	   (1 4 0 0 4 1)
	   (1 0 2 2 0 1)
	   (1 2 0 1 0 1)
	   (1 3 4 0 4 1)
	   (1 1 1 1 1 1)))

;(34)
(setq p9 '((1 1 1 1 1 1 1 1 1) 
	   (1 1 1 0 0 1 1 1 1) 
	   (1 0 0 0 0 0 2 0 1) 
	   (1 0 1 0 0 1 2 0 1) 
	   (1 0 4 4 4 1 3 0 1) 
	   (1 1 1 1 1 1 1 1 1)))

;(59)
(setq p10 '((1 1 1 1 1 0 0)
	    (1 4 0 0 1 1 0)
	    (1 3 2 0 0 1 1)
	    (1 1 0 2 0 0 1)
	    (0 1 1 0 2 0 1)
	    (0 0 1 1 0 0 1)
	    (0 0 0 1 1 4 1)
	    (0 0 0 0 1 4 1)
	    (0 0 0 0 1 4 1)
	    (0 0 0 0 1 1 1)))

;(?) (51)
(setq p11 '((0 0 1 0 0 0 0)
	    (0 2 1 4 0 4 0)
	    (0 2 0 4 0 0 0)	   
	    (3 2 1 1 1 4 0)
	    (0 0 1 4 0 0 0)))

;(?) (41)
(setq p12 '((1 1 1 1 1 0 0 0)
	    (1 0 0 4 1 0 0 0)
	    (1 2 1 0 1 1 1 1)
	    (1 4 0 0 0 0 0 1)
	    (1 0 0 5 0 5 0 1)
	    (1 0 5 0 1 0 1 1)
	    (1 1 1 0 3 0 1 0)
	    (0 0 1 1 1 1 1 0)))

;(?) (78)
(setq p13 '((1 1 1 1 1 1 1 1 1 1)
	    (1 3 0 0 1 0 0 4 4 1)
	    (1 0 2 0 2 0 0 4 4 1)
	    (1 0 2 2 2 1 1 4 4 1)
	    (1 0 0 0 0 1 1 4 4 1)
	    (1 1 1 1 1 1 0 0 0 0)))

;(?) (26)
(setq p14 '((0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0)
	    (0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0)
	    (1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1)
	    (0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0)
	    (0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0)
	    (0 0 0 0 0 0 0 0 3 0 0 0 0 0 0 0)
	    (0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0)
	    (0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0)
	    (1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1)
	    (0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0)
	    (0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0)
	    (0 0 0 0 1 0 0 0 0 0 4 1 0 0 0 0)
	    (0 0 0 0 1 0 2 0 0 0 4 1 0 0 0 0)	    
	    (0 0 0 0 1 0 2 0 0 0 4 1 0 0 0 0)
	    ))

;(?)
(setq p15 '((0 0 1 1 1 1 1 1 1 0)
	    (1 1 1 0 0 1 1 1 1 0)
	    (1 0 0 2 0 0 0 1 1 0)
	    (1 3 2 0 2 0 0 0 1 0)
	    (1 1 0 2 0 2 0 0 1 0)
	    (0 1 1 0 2 0 2 0 1 0)
	    (0 0 1 1 0 2 4 0 1 0)
	    (0 0 0 1 1 1 1 0 1 0)
	    (0 0 0 0 1 4 1 0 0 1)
	    (0 0 0 0 1 4 4 4 0 1)
	    (0 0 0 0 1 0 1 4 0 1)
	    (0 0 0 0 1 4 4 4 0 1)
	    (0 0 0 0 1 1 1 1 1 1)))

;(219)
(setq q1 '(
        (0 1 1 1 1 0 0 0)
	    (0 1 0 4 1 1 1 1)
	    (1 1 4 0 0 0 4 1)
	    (1 4 4 0 1 1 0 1)
	    (1 0 1 0 1 1 0 1)
	    (1 0 2 0 0 2 0 1)
	    (1 1 0 2 2 1 1 1)
	    (0 1 3 0 0 1 0 0)
	    (0 1 1 1 1 1 0 0)
	    ))
;(16)	    
(setq q2 '(
        (1 1 1 1 1 1 1)
	    (1 4 3 0 1 4 1)
	    (1 2 5 0 2 0 1)
	    (1 0 0 0 2 0 1)
	    (1 0 4 4 0 0 1)
	    (1 0 0 5 0 0 1)
	    (1 1 1 1 1 1 1)
	    ))
;(28)	    
(setq q3 '(
        (1 1 1 1 1 1 1)
	    (1 4 0 0 0 0 1)
	    (1 2 5 0 1 0 1)
	    (1 4 0 0 2 5 1)
	    (1 0 4 2 0 0 1)
	    (1 6 0 5 0 0 1)
	    (1 1 1 1 1 1 1)
	    ))
	    
;(28)	    
(setq q4 '(
        (1 1 1 1 1 1 1 1)
	    (1 6 0 0 1 1 0 1)
	    (1 0 2 2 0 0 0 1)
	    (1 0 5 4 0 4 0 1)
	    (1 0 2 0 5 2 4 1)
	    (1 4 0 4 0 2 0 1)
	    (1 1 1 0 0 5 0 1)
	    (1 1 1 1 1 1 1 1)
	    ))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

#|
 | Utility functions for printing states and moves.
 | You do not need to understand any of the functions below this point.
 |#

;
; Helper function of prettyMoves
; from s1 --> s2
;
(defun detectDiff (s1 s2)
  (let* ((k1 (getKeeperPosition s1 0))
	 (k2 (getKeeperPosition s2 0))
	 (deltaX (- (car k2) (car k1)))
	 (deltaY (- (cadr k2) (cadr k1)))
	 )
    (cond ((= deltaX 0) (if (> deltaY 0) 'DOWN 'UP))
	  (t (if (> deltaX 0) 'RIGHT 'LEFT))
	  );end cond
    );end let
  );end defun

;
; Translates a list of states into a list of moves.
; Usage: (prettyMoves (a* <problem> #'goal-test #'next-states #'heuristic))
;
(defun prettyMoves (m)
  (cond ((null m) nil)
	((= 1 (length m)) (list 'END))
	(t (cons (detectDiff (car m) (cadr m)) (prettyMoves (cdr m))))
	);end cond
  );

;
; Print the content of the square to stdout.
;
(defun printSquare (s)
  (cond ((= s blank) (format t " "))
	((= s wall) (format t "#"))
	((= s box) (format t "$"))
	((= s keeper) (format t "@"))
	((= s star) (format t "."))
	((= s boxstar) (format t "*"))
	((= s keeperstar) (format t "+"))
	(t (format t "|"))
	);end cond
  )

;
; Print a row
;
(defun printRow (r)
  (dolist (cur r)
    (printSquare cur)    
    )
  );

;
; Print a state
;
(defun printState (s)
  (progn    
    (dolist (cur s)
      (printRow cur)
      (format t "~%")
      )
    );end progn
  )

;
; Print a list of states with delay.
;
(defun printStates (sl delay)
  (dolist (cur sl)
    (printState cur)
    (sleep delay)
    );end dolist
  );end defun
  





;;;testcases
;;;goal-test
;(write-line "goal-test testcases")                                (FRESH-LINE) 
;(write (cond ((equal (goal-test p1) NIL) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (goal-test p2) NIL) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (goal-test p3) NIL) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (goal-test p4) NIL) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (goal-test p5) NIL) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (goal-test p6) NIL) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (goal-test p7) NIL) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (goal-test p8) NIL) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (goal-test p9) NIL) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (goal-test p10) NIL) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (goal-test p11) NIL) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (goal-test p12) NIL) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (goal-test p13) NIL) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (goal-test p14) NIL) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (goal-test p15) NIL) T) (t NIL)))         (FRESH-LINE)
;
;;;get-square
;(write-line "get-square testcases")                                (FRESH-LINE) 
;(write (cond ((equal (get-square p1 1 1) 0) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (get-square p1 2 2) 2) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (get-square p1 3 3) 1) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (get-square p1 10 10) 1) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (get-square p1 -1 1) 1) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (get-square p2 1 0) 1) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (get-square p2 2 1) 0) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (get-square p2 -1 0) 1) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (get-square p2 6 0) 1) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (get-square p2 1 4) 3) T) (t NIL)))         (FRESH-LINE)
;
;;;set-square
;(write-line "set-square testcases")                                (FRESH-LINE)
;(write (cond ((equal (get-square (set-square p1 1 1 5) 1 1) 5) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (get-square (set-square p2 3 2 4) 3 2) 4) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (get-square (set-square p3 0 2 0) 0 2) 0) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (get-square (set-square p4 3 0 3) 3 0) 3) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (get-square (set-square p5 2 2 2) 2 2) 2) T) (t NIL)))         (FRESH-LINE)
;
;;;next-states
;(write-line "next-states testcases")
;(setq s1 '((1 1 1 1 1)
;(1 4 0 0 1)
;(1 0 2 0 1)
;(1 0 3 0 1)
;(1 0 0 0 1)
;(1 1 1 1 1)
;))
;(setq s2 '((1 1 1 1 1)
;(1 0 0 4 1)
;(1 0 2 3 1)
;(1 0 0 0 1)
;(1 0 0 4 1)
;(1 1 1 1 1)
;))
;(setq s3 '((1 1 1 1 1)
;(1 0 0 6 1)
;(1 0 2 0 1)
;(1 0 0 0 1)
;(1 4 0 4 1)
;(1 1 1 1 1)
;))
;(setq s4 '((1 1 1 1 1)
;(1 0 2 4 1)
;(1 0 0 0 1)
;(1 0 0 0 1)
;(1 0 5 3 1)
;(1 1 1 1 1)
;))
;
;
;(write (cond 
;            ((equal (next-states s1)
;'(((1 1 1 1 1) (1 4 2 0 1) (1 0 3 0 1) (1 0 0 0 1) (1 0 0 0 1) (1 1 1 1 1))
;((1 1 1 1 1) (1 4 0 0 1) (1 0 2 0 1) (1 0 0 3 1) (1 0 0 0 1) (1 1 1 1 1)) 
;((1 1 1 1 1) (1 4 0 0 1) (1 0 2 0 1) (1 0 0 0 1) (1 0 3 0 1) (1 1 1 1 1)) 
;((1 1 1 1 1) (1 4 0 0 1) (1 0 2 0 1) (1 3 0 0 1) (1 0 0 0 1) (1 1 1 1 1))) 
;                ) T) 
;            (t NIL)
;        )
;            
;)         (FRESH-LINE)
;
;(write (cond 
;            ((equal (next-states s2)
;'(((1 1 1 1 1) (1 0 0 6 1) (1 0 2 0 1) (1 0 0 0 1) (1 0 0 4 1) (1 1 1 1 1))
;((1 1 1 1 1) (1 0 0 4 1) (1 0 2 0 1) (1 0 0 3 1) (1 0 0 4 1) (1 1 1 1 1))
;((1 1 1 1 1) (1 0 0 4 1) (1 2 3 0 1) (1 0 0 0 1) (1 0 0 4 1) (1 1 1 1 1))) 
;                ) T) 
;            (t NIL)
;        )
;            
;)         (FRESH-LINE)
;
;
;(write (cond 
;            ((equal (next-states s3)
;'(((1 1 1 1 1) (1 0 0 4 1) (1 0 2 3 1) (1 0 0 0 1) (1 4 0 4 1) (1 1 1 1 1))
;((1 1 1 1 1) (1 0 3 4 1) (1 0 2 0 1) (1 0 0 0 1) (1 4 0 4 1) (1 1 1 1 1)))
;                ) T) 
;            (t NIL)
;        )
;            
;)         (FRESH-LINE)
;
;
;(write (cond 
;            ((equal (next-states s4)
;'(((1 1 1 1 1) (1 0 2 4 1) (1 0 0 0 1) (1 0 0 3 1) (1 0 5 0 1) (1 1 1 1 1))
;((1 1 1 1 1) (1 0 2 4 1) (1 0 0 0 1) (1 0 0 0 1) (1 2 6 0 1) (1 1 1 1 1)))
;                ) T) 
;            (t NIL)
;        )
;            
;)         (FRESH-LINE)
;
;
;;;list-boxes-pos
;(write-line "list-boxes-pos testcases")
;(write (list-boxes-pos p1 0))                      (FRESH-LINE)
;(write (list-boxes-pos p2 0))                      (FRESH-LINE)
;(write (list-boxes-pos p3 0))                      (FRESH-LINE)
;(write (list-boxes-pos p4 0))                      (FRESH-LINE)
;(write (list-boxes-pos p5 0))                      (FRESH-LINE)
;(write (list-boxes-pos p6 0))                      (FRESH-LINE)
;(write (list-boxes-pos p7 0))                      (FRESH-LINE)
;(write (list-boxes-pos p8 0))                      (FRESH-LINE)
;(write (list-boxes-pos p9 0))                      (FRESH-LINE)
;(write (list-boxes-pos p10 0))                      (FRESH-LINE)
;(write (list-boxes-pos p11 0))                      (FRESH-LINE)
;(write (list-boxes-pos p12 0))                      (FRESH-LINE)
;(write (list-boxes-pos p13 0))                      (FRESH-LINE)
;(write (list-boxes-pos p14 0))                      (FRESH-LINE)
;(write (list-boxes-pos p15 0))                      (FRESH-LINE)
;
;
;;;list-goals-pos
;(write-line "list-goals-pos testcases")
;(write (list-goals-pos p1 0))                      (FRESH-LINE)
;(write (list-goals-pos p2 0))                      (FRESH-LINE)
;(write (list-goals-pos p3 0))                      (FRESH-LINE)
;(write (list-goals-pos p4 0))                      (FRESH-LINE)
;(write (list-goals-pos p5 0))                      (FRESH-LINE)
;(write (list-goals-pos p6 0))                      (FRESH-LINE)
;(write (list-goals-pos p7 0))                      (FRESH-LINE)
;(write (list-goals-pos p8 0))                      (FRESH-LINE)
;(write (list-goals-pos p9 0))                      (FRESH-LINE)
;(write (list-goals-pos p10 0))                      (FRESH-LINE)
;(write (list-goals-pos p11 0))                      (FRESH-LINE)
;(write (list-goals-pos p12 0))                      (FRESH-LINE)
;(write (list-goals-pos p13 0))                      (FRESH-LINE)
;(write (list-goals-pos p14 0))                      (FRESH-LINE)
;(write (list-goals-pos p15 0))                      (FRESH-LINE)
;(write-line "done")                               (FRESH-LINE) 

