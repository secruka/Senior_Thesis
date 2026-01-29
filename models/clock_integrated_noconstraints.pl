%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% models/clock_integrated_noconstraint.pl
%
% Same neural predicates.
% Keep rotation correction.
% Remove hour-minute dependency constraint (no minute-based hour shift).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nn(net_dial,   [X], R,  [0,1,2,3]) :: dial(X,R).
nn(net_hour,   [X], H0, [0,1,2,3,4,5,6,7,8,9,10,11]) :: hour_img(X,H0).
nn(net_minute, [X], M0, [0,1,2,3,4,5,6,7,8,9,10,11]) :: minute_img(X,M0).

rot_steps(0, 0).
rot_steps(1, 3).
rot_steps(2, 6).
rot_steps(3, 9).

correct_idx(ImageIdx, Steps, CanonIdx) :-
    CanonIdx is (ImageIdx - Steps + 120) mod 12.

idx_to_hour(0, 12).
idx_to_hour(I, I) :- I >= 1, I =< 11.

idx_to_minute(MIdx, Minute) :-
    Minute is MIdx * 5.

valid_time(Hour, Minute) :-
    Hour >= 1, Hour =< 12,
    Minute >= 0, Minute =< 55,
    0 is Minute mod 5.

% time without the hour-minute coupling constraint
time(X, Hour, Minute) :-
    dial(X, RIdx),
    hour_img(X, HImg),
    minute_img(X, MImg),

    rot_steps(RIdx, Steps),
    correct_idx(HImg, Steps, HPos),
    correct_idx(MImg, Steps, MIdx),

    idx_to_hour(HPos, Hour),
    idx_to_minute(MIdx, Minute),

    valid_time(Hour, Minute).
