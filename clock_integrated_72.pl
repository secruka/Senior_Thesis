%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% models/clock_integrated_72.pl
%
% DeepProbLog integration (dial 4-class + minute 12-class + short-hand 72-class)
%
% Goal predicate (supervision target):
%   time(Image, Hour, Minute).
%
% Key idea:
%   - minute_img is 12-class in IMAGE coords (5-min ticks): 0..11
%   - short72_img is 72-class in IMAGE coords (5-degree bins): 0..71
%   - dial is 4-class rotation (0,90,180,270) CLOCKWISE: 0..3
%   - We correct minute/short bins by dial rotation to CANONICAL coords.
%   - Then we enforce the clock relation between Minute and the short-hand angle.
%
% About 5° bins vs true short-hand motion:
%   With minute in 5-min steps, true short-hand advances 2.5° per step.
%   That lies between 5° bins for odd minute indices, so we allow the two nearest bins.
%   Additionally, we allow a small tolerance (±1 bin = 5°) via near72/2.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- neural predicates ---
nn(net_dial,   [X], R, [0,1,2,3]) :: dial(X,R).

% Short-hand: 72 bins (5° each) in IMAGE coordinates.
nn(net_hour,   [X], S0, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71]) :: hour_img(X,S0).

% Minute-hand: 12 bins (5-min ticks) in IMAGE coordinates.
nn(net_minute, [X], M0, [0,1,2,3,4,5,6,7,8,9,10,11]) :: minute_img(X,M0).

% Convenience predicate for pretraining hands
hands(X,S0,M0) :- hour_img(X,S0), minute_img(X,M0).

% --- rotation mapping: RIdx -> step offsets ---
% For minute (12 bins): 90° = 3 steps (30° each)
rot_steps12(0, 0).  % 0 deg
rot_steps12(1, 3).  % 90 deg
rot_steps12(2, 6).  % 180 deg
rot_steps12(3, 9).  % 270 deg

% For short-hand (72 bins): 90° = 18 steps (5° each)
rot_steps72(0, 0).   % 0 deg
rot_steps72(1, 18).  % 90 deg
rot_steps72(2, 36).  % 180 deg
rot_steps72(3, 54).  % 270 deg

% --- safe modulo correction (avoid negatives) ---
correct_idx12(ImageIdx, Steps, CanonIdx) :-
    CanonIdx is (ImageIdx - Steps + 120) mod 12.

correct_idx72(ImageIdx, Steps, CanonIdx) :-
    CanonIdx is (ImageIdx - Steps + 7200) mod 72.

% --- decode indices to time values ---
idx_to_hour(0, 12).
idx_to_hour(I, I) :- I >= 1, I =< 11.

idx_to_minute(MIdx, Minute) :-
    Minute is MIdx * 5.

valid_time(Hour, Minute) :-
    Hour >= 1, Hour =< 12,
    Minute >= 0, Minute =< 55,
    0 is Minute mod 5.

% --- circular distance on 72-cycle ---
circ_dist72(A, B, D) :-
    D1 is abs(A - B),
    D2 is 72 - D1,
    ( D1 =< D2 -> D is D1 ; D is D2 ).

% allow ±1 bin (5°) tolerance
near72(A, B) :-
    circ_dist72(A, B, D),
    D =< 1.

% --- expected short-hand bin from (HourIdx0, MinuteIdx12) ---
% HourIdx0: 0..11 where 0 means 12 oclock
% MinuteIdx12: 0..11 where each step is 5 minutes
%
% True short-hand angle (deg) = 30*HourIdx0 + 2.5*MinuteIdx12  (mod 360)
% Convert to 5° bins: divide by 5 -> 6*HourIdx0 + 0.5*MinuteIdx12
%
% If MinuteIdx12 is even, 0.5*MinuteIdx12 is integer -> unique expected bin.
% If odd, it lies exactly between two bins -> allow both neighbors.
short_expected72(HIdx0, MIdx12, SExp) :-
    0 is MIdx12 mod 2,
    Off is MIdx12 // 2,
    SExp is (6*HIdx0 + Off) mod 72.

short_expected72(HIdx0, MIdx12, SExp) :-
    1 is MIdx12 mod 2,
    Off is MIdx12 // 2,
    SExp is (6*HIdx0 + Off) mod 72.

short_expected72(HIdx0, MIdx12, SExp) :-
    1 is MIdx12 mod 2,
    Off is MIdx12 // 2,
    SExp is (6*HIdx0 + Off + 1) mod 72.

% --- main integration ---
% time(Image, Hour, Minute)
time(X, Hour, Minute) :-
    dial(X, RIdx),
    hour_img(X, SImg),
    minute_img(X, MImg),

    rot_steps72(RIdx, Steps72),
    rot_steps12(RIdx, Steps12),

    correct_idx72(SImg, Steps72, SCanon),
    correct_idx12(MImg, Steps12, MIdx12),

    idx_to_minute(MIdx12, Minute),

    % choose HourIdx0 (0..11) that matches the corrected short-hand bin given MIdx12
    member(HIdx0, [0,1,2,3,4,5,6,7,8,9,10,11]),
    short_expected72(HIdx0, MIdx12, SExp),
    near72(SCanon, SExp),

    idx_to_hour(HIdx0, Hour),
    valid_time(Hour, Minute).
