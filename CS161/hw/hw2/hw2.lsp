;;;; CS161A Asignment 2 Solution
;;;; Name: Ashish Kumar Singh
;;;; UID: 105479019


;;; If first element of TREE is a leaf node, then perform BFS on remainning Tree and append the first node on front. Else append the childrens of first node to the back of the list and recursively call BFS on the whole tree
(defun BFS (TREE)
    ;;Input: a TREE represented as lists in which a leaf node is represented by an atom, and a non-leaf node is represented by a list of its child nodes.
    ;;Output: a single, top-level list of the terminal nodes in the order they would be visited by a left-to-right breadth-first search
    (cond 
        ((null TREE)NIL)
        ((atom TREE)(list TREE))
        ((= 1 (length TREE)) (BFS (car TREE)))
        ((atom (car TREE)) (append (list (car TREE)) (BFS (cdr TREE))))
        
        (t (BFS (append (cdr TREE)(car TREE))))
    )
)

;;; If first element of TREE is a leaf node, then perform DFS on remainning Tree and append the first node on front. Else append the childrens of first node to the front of the list and recursively call DFS on the whole tree
(defun DFS (TREE)
    ;;Input: a TREE represented as lists in which a leaf node is represented by an atom, and a non-leaf node is represented by a list of its child nodes.
    ;;Output: a single, top-level list of the terminal nodes in the order they would be visited by a left-to-right depth-first search
    (cond 
        ((null TREE)NIL)
        ((atom TREE)(list TREE))
        ((= 1 (length TREE)) (DFS (car TREE)))
        ((atom (car TREE)) (append (list (car TREE)) (DFS (cdr TREE))))
        
        (t (DFS (append (car TREE) (cdr TREE))))
    )
)

;;;recursively append atoms than are at depth less than M and removes all others
(defun TREETRIM (TREE M)
    ;;Input: list representation of a TREE and a number M
    ;;Output: a TREE in which all the nodes with depth greater than M are removed
    (cond
        ((null TREE)NIL)
        ((= M 0)NIL)
        ((atom TREE) TREE)
        ((atom (car TREE)) (cons (car TREE) (TREETRIM (cdr TREE) M) ))
        (t ( let
                (   
                    (exp1 (TREETRIM (car TREE) (- M 1) ))
                )
                (if (null exp1) (TREETRIM (cdr TREE) M) (cons exp1 (TREETRIM (cdr TREE) M)) )
            )
        )
    )

)

;;;recursively reverse a list
(defun REV (L)
    ;;Input: list L
    ;;Output: list with the atoms of L in revverse order
    (cond
        ((null L)NIL)
        (t (append (REV (cdr L)) (list (car L)) ))
    )
)

;;;we recursively call dfid function for a lower max depth and append it with the reverse fo the dfs done on the tree trimmed with M depth
(defun DFID (TREE M)
    ;;Input: TREE with list representation and a number M representing the maximum depth of the tree
    ;;Output: A single top-level list of the terminal nodes in the order that they would be visited by a right-to-left depth-first iterative-deepening search
    (cond
        ((null TREE)NIL)
        ((AND (atom TREE) (= M 0))(list TREE ))
        ((= M 0)NIL)
        (t (let
                (
                    (TrimmedTree (TREETRIM TREE M))
                )
                (append (DFID TREE (- M 1)) (REV (DFS TrimmedTree)) )
            )
        )
    )
)




; These functions implement a depth-first solver for the River-Boat
; problem. In this problem, three members from Group-X, denoted XXX,
; and three members from Group-O, denoted OOO, are trying to
; go from the east side of a river to the west side. They have a single boat
; that can carry two people at a time from one side of the river to the
; other. There must be at least one person in the boat to cross the river. There
; can never be more O's on one side of the river than X's.

; In this implementation, a state is represented by a single list
; (#X #O side). side represents which side the boat is
; currently on, and is T if it is on the east side and NIL if on the west
; side. #X and #O represent the number of X's and
; O's on the same side as the boat. Thus, the initial state for this
; problem is (3 3 T) (three X's, three O's, and the boat are all
; on the east side of the river) and the goal state is (3 3 NIL).

; The main entry point for this solver is the function MC-DFS, which is called
; with the initial state to search from and the path to this state. It returns
; the complete path from the initial state to the goal state: this path is a
; list of intermediate problem states. The first element of the path is the
; initial state and the last element is the goal state. Each intermediate state
; is the state that results from applying the appropriate operator to the
; preceding state. If there is no solution, MC-DFS returns NIL.

; To call MC-DFS to solve the original problem, one would call (MC-DFS '(3 3 T)
; NIL) -- however, it would be possible to call MC-DFS with a different initial
; state or with an initial path.

; Examples of calls to some of the helper functions can be found after the code.



; FINAL-STATE takes a single argument s, the current state, and returns T if it
; is the goal state (3 3 NIL) and NIL otherwise.
(defun final-state (s)
    (let
        (   
            (goal '(3 3 NIL))
        )
        (if (equal s goal) T NIL)
    )
)

; NEXT-STATE returns the state that results from applying an operator to the
; current state. It takes three arguments: the current state (s), a number of
; X's to move (m), and a number of O's to move (c). It returns a
; list containing the state that results from moving that number of X's
; and O's from the current side of the river to the other side of the
; river. If applying this operator results in an invalid state (because there
; are more O's than X's on either side of the river, or because
; it would move more X's or O's than are on this side of the
; river) it returns NIL.
;
; NOTE that next-state returns a list containing the successor state (which is
; itself a list); the return should look something like ((1 1 T)).
(defun next-state (s m c)
    (let*
        (
            (numx (car s))
            (numo (cadr s))
            (nump (+ m c))
            (side (caddr s))
            (newx (- numx m))
            (newo (- numo c))
            (otherx (- 3 newx))
            (othero (- 3 newo))
        )
        (cond 
            ((OR (< newx 0) (< m 0) (< newo 0) (< c 0) (> nump 2) (= nump 0))NIL)
            ((AND (> newo newx) (not (= newx 0)) )NIL)
            ((AND (> othero otherx) (not (= otherx 0)) )NIL)
            (t (list (list otherx othero (not side) )))
        )
    )
)

; SUCC-FN returns all of the possible legal successor states to the current
; state. It takes a single argument (s), which encodes the current state, and
; returns a list of each state that can be reached by applying legal operators
; to the current state.
(defun succ-fn (s)
    (let
        (
            (ns1 (next-state s 0 1))
            (ns2 (next-state s 0 2))
            (ns3 (next-state s 1 0))
            (ns4 (next-state s 2 0))
            (ns5 (next-state s 1 1))
        )
        (append ns1 ns2 ns3 ns4 ns5)
    )
)

; ON-PATH checks whether the current state is on the stack of states visited by
; this depth-first search. It takes two arguments: the current state (s) and the
; stack of states visited by MC-DFS (states). It returns T if s is a member of
; states and NIL otherwise.
(defun on-path (s states)
    (cond
        ((null states)NIL)
        ((equal s (car states))t)
        (t (on-path s (cdr states) ))
    )
)

; MULT-DFS is a helper function for MC-DFS. It takes two arguments: a stack of
; states from the initial state to the current state (path), and the legal
; successor states from the current state (states).
; MULT-DFS does a depth-first search on each element of states in
; turn. If any of those searches reaches the final state, MULT-DFS returns the
; complete path from the initial state to the goal state. Otherwise, it returns
; NIL. 
; Note that the path should be ordered as: (S_n ... S_2 S_1 S_0)
(defun mult-dfs (states path)
    (cond
        ((null states)NIL)
        (t (let*
                (
                    (s (car states))
                    (check (mc-dfs s path ))
                )
                (if (null check) (mult-dfs (cdr states) path) check)
        
        
            )
        )
    )
)


; MC-DFS does a depth first search from a given state to the goal state. It
; takes two arguments: a state (S) and the path from the initial state to S
; (PATH). If S is the initial state in our search, PATH should be NIL. MC-DFS
; performs a depth-first search starting at the given state. It returns the path
; from the initial state to the goal state, if any, or NIL otherwise. MC-DFS is
; responsible for checking if S is already the goal state, as well as for
; ensuring that the depth-first search does not revisit a node already on the
; search path.
(defun mc-dfs (s path)
    (cond
        ((final-state s) (append (list s) path) )
        ((null path) (mult-dfs (succ-fn s)(list s)) )
        ((on-path s path)NIL)
        (t (mult-dfs (succ-fn s) (append (list s) path)))
    )
)


;;testcases
;;TREETRIM
;(write-line "TREETRIM testcases")
;(write (cond ((equal (TREETRIM '((A (B)) C (D)) 1) '(C)) T) (t NIL)))                   (FRESH-LINE)
;(write (cond ((equal (TREETRIM '((A (B)) C (D)) 2) '((A) C (D))) T) (t NIL)))           (FRESH-LINE)
;(write (cond ((equal (TREETRIM '((A (B)) C (D)) 3) '((A (B)) C (D))) T) (t NIL)))       (FRESH-LINE)
;
;;BFS
;(write-line "BFS testcases")
;(write (cond ((equal (BFS '()) NIL) T) (t NIL)))                            (FRESH-LINE)
;(write (cond ((equal (BFS '1) '(1)) T) (t NIL)))                            (FRESH-LINE)
;(write (cond ((equal (BFS '((A (B)) C (D))) '(C A D B)) T) (t NIL)))        (FRESH-LINE)
;(write (cond ((equal (BFS '( C (D))) '(C D)) T) (t NIL)))                           (FRESH-LINE)
;(write (cond ((equal (BFS '((A B) (C D) )) '(A B C D)) T) (t NIL)))                 (FRESH-LINE)
;(write (cond ((equal (BFS '(A(B C) (D) (E (F G)))) '(A B C D E F G)) T) (t NIL)))        (FRESH-LINE)
;(write (cond ((equal (BFS '((A B) (C) (D (E) F) G)) '(G A B C D F E)) T) (t NIL)))        (FRESH-LINE)
;
;;DFS
;(write-line "DFS testcases")
;(write (cond ((equal (DFS '()) NIL) T) (t NIL)))                            (FRESH-LINE)
;(write (cond ((equal (DFS '1) '(1)) T) (t NIL)))                            (FRESH-LINE)
;(write (cond ((equal (DFS '((A (B)) C (D))) '(A B C D)) T) (t NIL)))        (FRESH-LINE)
;(write (cond ((equal (DFS '( C (D))) '(C D)) T) (t NIL)))                           (FRESH-LINE)
;(write (cond ((equal (DFS '((A B) (C D) )) '(A B C D)) T) (t NIL)))                 (FRESH-LINE)
;(write (cond ((equal (DFS '(A(B C) (D) (E (F G)))) '(A B C D E F G)) T) (t NIL)))        (FRESH-LINE)
;(write (cond ((equal (DFS '((A B) (C) (D (E) F) G)) '(A B C D E F G)) T) (t NIL)))        (FRESH-LINE)
;
;;DFID
;(write-line "DFID testcases")
;(write (cond ((equal (DFID '() 2) NIL) T) (t NIL)))                             (FRESH-LINE)
;(write (cond ((equal (DFID '1 1) '(1 1)) T) (t NIL)))                            (FRESH-LINE)
;(write (cond ((equal (DFID '1 0) '(1)) T) (t NIL)))                             (FRESH-LINE)
;(write (cond ((equal (DFID '(A B) 2) '(B A B A)) T) (t NIL)))                            (FRESH-LINE)
;(write (cond ((equal (DFID '((A (B)) C (D)) 3) '(C D C A D C B A)) T) (t NIL)))         (FRESH-LINE)
;(write (cond ((equal (DFID '(A(B C) (D) (E (F G))) 3) '(A E D C B A G F E D C B A)) T) (t NIL)))        (FRESH-LINE)
;(write (cond ((equal (DFID '((A B) (C) (D (E) F) G) 3) '(G G F D C B A G F E D C B A)) T) (t NIL)))        (FRESH-LINE)
;
;;next-state
;(write-line "next-state testcases")
;(write (cond ((equal (next-state '(3 3 t) 1 0) NIL) T) (t NIL)))                    (FRESH-LINE)
;(write (cond ((equal (next-state '(3 3 t) 0 1) '((0 1 NIL))) T) (t NIL)))           (FRESH-LINE)
;(write (cond ((equal (next-state '(3 3 t) 0 3) '()) T) (t NIL)))           (FRESH-LINE)
;(write (cond ((equal (next-state '(3 3 t) 1 1) '((1 1 NIL))) T) (t NIL)))           (FRESH-LINE)
;(write (cond ((equal (next-state '(2 2 t) 0 1) '()) T) (t NIL)))           (FRESH-LINE)
;
;;succ-fn
;(write-line "succ-fn testcases")
;(write (cond ((equal (succ-fn '(3 3 t)) '((0 1 NIL) (0 2 NIL) (1 1 NIL))) T) (t NIL)))       (FRESH-LINE)
;(write (cond ((equal (succ-fn '(2 2 NIL)) '((3 1 T) (2 2 T))) T) (t NIL)))                    (FRESH-LINE)
;(write (cond ((equal (succ-fn '(1 1 t)) '((3 2 NIL) (3 3 NIL))) T) (t NIL)))                    (FRESH-LINE)
;(write (cond ((equal (succ-fn '(3 1 t)) '((0 3 NIL) (2 2 NIL))) T) (t NIL)))                    (FRESH-LINE)
;
;;mult-dfs
;(write-line "mult-dfs/mc-dfs testcases")
;(write (cond ((equal (mc-dfs '(3 3 t) '((3 3 t))) '()) T) (t NIL)))                    (FRESH-LINE)
;(write (cond ((equal (mc-dfs '(3 3 nil) '((3 3 nil))) '((3 3 NIL) (3 3 NIL))) T) (t NIL)))                    (FRESH-LINE;)
;(write (cond ((equal (mc-dfs '(3 3 nil) '()) '((3 3 NIL))) T) (t NIL)))                    (FRESH-LINE)
;
;(write (mult-dfs (succ-fn '(3 3 t)) '((3 3 t))) )               (FRESH-LINE)
;(write (mc-dfs '(3 3 t)  '() ))                                 (FRESH-LINE)
;(write (mc-dfs '(1 2 t)  '() ))                                 (FRESH-LINE)
;(write (mc-dfs '(1 1 NIL)  '((3 3 t)) ))                        (FRESH-LINE)
;
;(write-line "Done")



