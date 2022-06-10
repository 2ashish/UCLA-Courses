kenken_testcase(
  6,
  [
   +(11, [[1|1], [2|1]]),
   /(2, [1|2], [1|3]),
   *(20, [[1|4], [2|4]]),
   *(6, [[1|5], [1|6], [2|6], [3|6]]),
   -(3, [2|2], [2|3]),
   /(3, [2|5], [3|5]),
   *(240, [[3|1], [3|2], [4|1], [4|2]]),
   *(6, [[3|3], [3|4]]),
   *(6, [[4|3], [5|3]]),
   +(7, [[4|4], [5|4], [5|5]]),
   *(30, [[4|5], [4|6]]),
   *(6, [[5|1], [5|2]]),
   +(9, [[5|6], [6|6]]),
   +(8, [[6|1], [6|2], [6|3]]),
   /(2, [6|4], [6|5])
  ]
).

%argument order change to use for maplist
list_length(N, L):- length(L, N).
list_domain(A, B, L):- fd_domain(L, A, B).

%constraint checking for -
check_constraint(T, -(S, [A|B], [C|D])):-
    nth(A, T, Row1),
    nth(B, Row1, X),
    nth(C, T, Row2),
    nth(D, Row2, Y),
    (S #= X - Y; S #= Y - X).

%constraint checking for /
check_constraint(T, /(S, [A|B], [C|D])):-
    nth(A, T, Row1),
    nth(B, Row1, X),
    nth(C, T, Row2),
    nth(D, Row2, Y),
    (S #= X / Y; S #= Y / X).

%constraint checking for +
check_constraint(T, +(S, A)):-
    check_sum(T,A,X),
    S #= X.

%constraint checking for *
check_constraint(T, *(S, A)):-
    check_mul(T,A,X),
    S #= X.

check_sum(_,[],0).
check_sum(T,[[A|B]|C],X):-
    nth(A, T, Row1),
    nth(B, Row1, Y),
    check_sum(T,C,Z),
    X #= Y + Z.

check_mul(_,[],1).
check_mul(T,[[A|B]|C],X):-
    nth(A, T, Row1),
    nth(B, Row1, Y),
    check_mul(T,C,Z),
    X #= Y * Z.

%transpose code from https://stackoverflow.com/questions/4280986/how-to-transpose-a-matrix-in-prologmaplist
transpose([], []).
transpose([F|Fs], Ts) :-
    transpose(F, [F|Fs], Ts).

transpose([], _, []).
transpose([_|Rs], Ms, [Ts|Tss]) :-
        lists_firsts_rests(Ms, Ts, Ms1),
        transpose(Rs, Ms1, Tss).

lists_firsts_rests([], [], []).
lists_firsts_rests([[F|Os]|Rest], [F|Fs], [Os|Oss]) :-
        lists_firsts_rests(Rest, Fs, Oss).

%kenken solver with fd primitives
kenken(N,C,T) :- 
    length(T,N),
    maplist(list_length(N),T),
    maplist(list_domain(1,N),T),
    maplist(fd_all_different,T),
    transpose(T,TT),
    maplist(fd_all_different,TT),
    maplist(check_constraint(T),C),
    maplist(fd_labeling,T).

% similar code for plain_kenken without fd primitives
list_between(A, B, L):- 
    maplist(between(A, B),L).

all_unique([]).
all_unique([H|T]) :-
    member(H, T), !, fail.
all_unique([_|T]) :- all_unique(T).

check_constraint_plain(T, -(S, [A|B], [C|D])):-
    nth(A, T, Row1),
    nth(B, Row1, X),
    nth(C, T, Row2),
    nth(D, Row2, Y),
    (S is X - Y; S is Y - X).

check_constraint_plain(T, /(S, [A|B], [C|D])):-
    nth(A, T, Row1),
    nth(B, Row1, X),
    nth(C, T, Row2),
    nth(D, Row2, Y),
    (S is X / Y; S is Y / X).

check_constraint_plain(T, +(S, A)):-
    check_sum_plain(T,A,X),
    S is X.

check_constraint_plain(T, *(S, A)):-
    check_mul_plain(T,A,X),
    S is X.

check_sum_plain(_,[],0).
check_sum_plain(T,[[A|B]|C],X):-
    nth(A, T, Row1),
    nth(B, Row1, Y),
    check_sum_plain(T,C,Z),
    X is Y + Z.

check_mul_plain(_,[],1).
check_mul_plain(T,[[A|B]|C],X):-
    nth(A, T, Row1),
    nth(B, Row1, Y),
    check_mul_plain(T,C,Z),
    X is Y * Z.

plain_kenken(N,C,T) :- 
    length(T,N),
    maplist(list_length(N),T),
    maplist(list_between(1,N),T),
    maplist(all_unique,T),
    transpose(T,TT),
    maplist(all_unique,TT),
    maplist(check_constraint_plain(T),C).

%using statistics shows kenken took 4ms while plain_kenken took 1798072ms for solving N=4 testcase

%API for no-op Kenken
%noop_kenken/4 accepts 4 args
%N, a nonnegative integer specifying the number of cells on each side of the KenKen square.
%C, a list of numeric cage constraints as described below.
%T, a list of list of integers. T and its members all have length N. This represents the N×N grid.
%O, a list of N integers mapping the operation for corresponding constraint {e.g. add:0, sub:1, mul:2, div:3}
%Each constraint in C is of the following form:
%(S,L): where L is a nonempty list, and the integer S is target for the elements in the list L of squares.
%
%On successful call, list of list T will contain the matrix with constraints satisfied, and list O will contain the operation assignmnet to constraints
%
%noop_kenken_testcase(
%  4,
%  [
%   (12, [[1|1], [1|2], [1|3]]),
%   (12, [[2|1], [3|1], [3|2], [4|1]]),
%   (12, [[1|4], [2|2], [2|3], [2|4], [3|3], [3|4]]),
%   (12, [[4|2], [4|3], [4|4]])
%  ]
%).
%
%run testcase as
%noop_kenken_testcase(N,C), noop_kenken(N,C,T,O).    
%   
%output would be:
%
%C = [
%   (12, [[1|1], [1|2], [1|3]]),
%   (12, [[2|1], [3|1], [3|2], [4|1]]),
%   (12, [[1|4], [2|2], [2|3], [2|4], [3|3], [3|4]]),
%   (12, [[4|2], [4|3], [4|4]])
%  ]
%N = 4
%T = [
%    [1,4,3,2],
%    [3,2,1,4],
%    [4,3,2,1],
%    [2,1,4,3]
%    ]
%O = [2,0,0,2] ?

