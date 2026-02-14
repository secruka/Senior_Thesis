%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% models/clock_latent.pl
%
% DeepProbLog program for latent-variable clock recognition.
%
% Supervision: time(Image, Hour, Minute) ONLY.
% The decomposition into dial rotation (R), short-hand angle (S),
% and minute-hand angle (M) is fully latent — DeepProbLog marginalises
% over all valid (R, S, M) groundings that satisfy the time constraint.
%
% Neural predicates (all outputs are latent — no direct labels):
%   net_rot    : dial rotation classifier     (4 classes: 0/90/180/270)
%   net_hour   : short-hand angle classifier  (72 classes, 5-deg bins, image coords)
%   net_minute : minute-hand angle classifier (12 classes, 30-deg bins, image coords)
%
% Key constraint:
%   The short-hand's canonical 5-deg bin is determined by BOTH hour AND minute
%   via short_expected72/3. This couples the hour and minute heads through
%   the clock's continuous geometry, making DeepProbLog's marginalisation
%   essential for learning.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- neural predicates (output softmax probabilities) ---
nn(net_rot,    [X], R, [0,1,2,3]) :: dial(X, R).
nn(net_hour,   [X], S0, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71]) :: hour_img(X, S0).
nn(net_minute, [X], M0, [0,1,2,3,4,5,6,7,8,9,10,11]) :: minute_img(X, M0).

% --- rotation -> step offsets ---
% For minute (12 bins): 90 deg = 3 steps of 30 deg
rot_steps12(0, 0).
rot_steps12(1, 3).
rot_steps12(2, 6).
rot_steps12(3, 9).

% For short-hand (72 bins): 90 deg = 18 steps of 5 deg
rot_steps72(0, 0).
rot_steps72(1, 18).
rot_steps72(2, 36).
rot_steps72(3, 54).

% --- modular correction: image coords -> canonical coords ---
correct_idx12(ImageIdx, Steps, CanonIdx) :-
    CanonIdx is (ImageIdx - Steps + 120) mod 12.

correct_idx72(ImageIdx, Steps, CanonIdx) :-
    CanonIdx is (ImageIdx - Steps + 7200) mod 72.

% --- decode indices to time values ---
idx_to_hour(0, 12).
idx_to_hour(I, I) :- I >= 1, I =< 11.

idx_to_minute(MIdx, Minute) :-
    Minute is MIdx * 5.

% --- time validity check ---
valid_time(Hour, Minute) :-
    Hour >= 1, Hour =< 12,
    Minute >= 0, Minute =< 55,
    0 is Minute mod 5.

% --- circular distance on 72-cycle ---
circ_dist72(A, B, D) :-
    D1 is abs(A - B),
    D2 is 72 - D1,
    D1 =< D2,
    D is D1.

circ_dist72(A, B, D) :-
    D1 is abs(A - B),
    D2 is 72 - D1,
    D1 > D2,
    D is D2.

% allow +/- 1 bin (5 deg) tolerance
near72(A, B) :-
    circ_dist72(A, B, D),
    D =< 1.

% --- expected short-hand 5-deg bin from (HourIdx0, MinuteIdx12) ---
%
% True short-hand angle (deg) = 30 * HourIdx0 + 2.5 * MinuteIdx12  (mod 360)
% In 5-deg bins: 6 * HourIdx0 + 0.5 * MinuteIdx12
%
% Even MinuteIdx12: 0.5 * M is integer -> unique expected bin.
% Odd  MinuteIdx12: lies between two bins -> allow both neighbours.
short_expected72(HIdx0, MIdx12, SExp) :-
    0 is MIdx12 mod 2,
    Off is MIdx12 // 2,
    SExp is (6 * HIdx0 + Off) mod 72.

short_expected72(HIdx0, MIdx12, SExp) :-
    1 is MIdx12 mod 2,
    Off is MIdx12 // 2,
    SExp is (6 * HIdx0 + Off) mod 72.

short_expected72(HIdx0, MIdx12, SExp) :-
    1 is MIdx12 mod 2,
    Off is MIdx12 // 2,
    SExp is (6 * HIdx0 + Off + 1) mod 72.

% --- sole query predicate: time(Image, Hour, Minute) ---
% R, S, M are all latent — marginalised by DeepProbLog.
time(X, Hour, Minute) :-
    dial(X, RIdx),
    hour_img(X, SImg),
    minute_img(X, MImg),
    rot_steps72(RIdx, Steps72),
    rot_steps12(RIdx, Steps12),
    correct_idx72(SImg, Steps72, SCanon),
    correct_idx12(MImg, Steps12, MIdx12),
    idx_to_minute(MIdx12, Minute),
    member(HIdx0, [0,1,2,3,4,5,6,7,8,9,10,11]),
    short_expected72(HIdx0, MIdx12, SExp),
    near72(SCanon, SExp),
    idx_to_hour(HIdx0, Hour),
    valid_time(Hour, Minute).
