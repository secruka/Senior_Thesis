%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% models/clock_integrated.pl
%
% DeepProbLog integration of:
%  - dial rotation classifier (4 classes: 0,90,180,270)
%  - hands angle classifiers (hour/minute, 12 classes each)
%
% Goal predicate (supervision target):
%   time(Image, Hour, Minute).
%
% Assumptions:
%   - dial(X,R) where R in {0,1,2,3} corresponds to rotation {0,90,180,270} degrees CLOCKWISE.
%   - hour_img(X,H0) and minute_img(X,M0) are hand-angle classes in IMAGE coordinates:
%       0=12 o'clock, 3=3 o'clock, 6=6 o'clock, 9=9 o'clock.
%   - We correct both hand indices by the dial rotation.
%   - Hour hand labels are nearest-tick (30°) quantization.
%     Therefore:
%       if minute<30 -> hour_time = hour_pos
%       else         -> hour_time = hour_pos - 1 (mod 12)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- +/-1 tolerance model (per image) ---
% Choose one delta per image X (sums to 1.0 => exactly one choice)
% Conservative setting (おすすめ): 0.15 / 0.70 / 0.15
0.15::err_h(X,-1); 0.70::err_h(X,0); 0.15::err_h(X,1).
0.15::err_m(X,-1); 0.70::err_m(X,0); 0.15::err_m(X,1).

% shift index on 12-cycle
shift_idx(I, D, J) :-
    J is (I + D + 120) mod 12.

% --- neural predicates ---
% This will be alrighty
nn(net_dial,   [X], R, [0,1,2,3]) :: dial(X,R).

nn(net_hour,   [X], H0, [0,1,2,3,4,5,6,7,8,9,10,11]) :: hour_img(X,H0).

nn(net_minute, [X], M0, [0,1,2,3,4,5,6,7,8,9,10,11]) :: minute_img(X,M0).

% Convenience predicate for pretraining hands
hands(X,H0,M0) :- hour_img(X,H0), minute_img(X,M0).

% --- rotation mapping: RIdx -> 12-step offset ---
rot_steps(0, 0).  % 0 deg
rot_steps(1, 3).  % 90 deg
rot_steps(2, 6).  % 180 deg
rot_steps(3, 9).  % 270 deg

% --- safe modulo correction (avoid negatives) ---
correct_idx(ImageIdx, Steps, CanonIdx) :-
    CanonIdx is (ImageIdx - Steps + 120) mod 12.

prev_idx(I, P) :-
    P is (I - 1 + 12) mod 12.

% --- decode indices to time values ---
idx_to_hour(0, 12).
idx_to_hour(I, I) :- I >= 1, I =< 11.

idx_to_minute(MIdx, Minute) :-
    Minute is MIdx * 5.

% Hour decoding with clock constraint
% minute idx 0..5  => 00..25 -> hour_time = hour_pos
hour_time_from_pos(HPos, MIdx, Hour) :-
    MIdx < 6,
    idx_to_hour(HPos, Hour).

% minute idx 6..11 => 30..55 -> hour_time = hour_pos - 1 (mod 12)
hour_time_from_pos(HPos, MIdx, Hour) :-
    MIdx >= 6,
    prev_idx(HPos, HTimeIdx),
    idx_to_hour(HTimeIdx, Hour).

valid_time(Hour, Minute) :-
    Hour >= 1, Hour =< 12,
    Minute >= 0, Minute =< 55,
    0 is Minute mod 5.

% --- main integration ---
% time(Image, Hour, Minute)
% integrates dial rotation + both hand predictions under clock constraints.
time(X, Hour, Minute) :-
    dial(X, RIdx),
    hour_img(X, HImg),
    minute_img(X, MImg),

    rot_steps(RIdx, Steps),

    % canonical indices (after rotation correction)
    correct_idx(HImg, Steps, HPos0),
    correct_idx(MImg, Steps, MIdx0),

    % +/-1 tolerance (probabilistic)
    err_h(X, DH),
    err_m(X, DM),
    shift_idx(HPos0, DH, HPos),
    shift_idx(MIdx0, DM, MIdx),

    idx_to_minute(MIdx, Minute),
    hour_time_from_pos(HPos, MIdx, Hour),

    valid_time(Hour, Minute).
