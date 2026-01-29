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
%
% New (soft) clock constraint:
%   - HourTimeIdx is either HPos (no shift) or prev(HPos) (shift -1),
%     chosen probabilistically depending on minute bin MIdx.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ----------------------------------------------------------------------
% 1) Neural predicates
% ----------------------------------------------------------------------

nn(net_dial,   [X], R,  [0,1,2,3]) :: dial(X,R).

nn(net_hour,   [X], H0, [0,1,2,3,4,5,6,7,8,9,10,11]) :: hour_img(X,H0).

nn(net_minute, [X], M0, [0,1,2,3,4,5,6,7,8,9,10,11]) :: minute_img(X,M0).

% Convenience predicate for pretraining hands (pure conjunction)
hands(X,H0,M0) :- hour_img(X,H0), minute_img(X,M0).

% ----------------------------------------------------------------------
% 2) Rotation mapping: RIdx -> 12-step offset (30° per step)
% ----------------------------------------------------------------------

rot_steps(0, 0).  % 0 deg
rot_steps(1, 3).  % 90 deg
rot_steps(2, 6).  % 180 deg
rot_steps(3, 9).  % 270 deg

% ----------------------------------------------------------------------
% 3) Index correction helpers
% ----------------------------------------------------------------------

% Safe modulo correction (avoid negatives)
correct_idx(ImageIdx, Steps, CanonIdx) :-
    CanonIdx is (ImageIdx - Steps + 120) mod 12.

% Previous index (mod 12)
prev_idx(I, P) :-
    P is (I - 1 + 12) mod 12.

% ----------------------------------------------------------------------
% 4) Decode indices to time values
% ----------------------------------------------------------------------

idx_to_hour(0, 12).
idx_to_hour(I, I) :- I >= 1, I =< 11.

idx_to_minute(MIdx, Minute) :-
    Minute is MIdx * 5.

valid_time(Hour, Minute) :-
    Hour >= 1, Hour =< 12,
    Minute >= 0, Minute =< 55,
    0 is Minute mod 5.

% ----------------------------------------------------------------------
% 5) Soft hour-time constraint (minute-dependent)
% ----------------------------------------------------------------------
% We choose hour-time index (HTimeIdx) probabilistically:
%   - "no shift": HTimeIdx = HPos
%   - "shift -1": HTimeIdx = prev(HPos)
%
% MIdx corresponds to minutes:
%   0..11 -> 0,5,10,15,20,25,30,35,40,45,50,55
%
% Intuition:
%   - For small minutes, hour label should usually be HPos (no shift).
%   - After 30+, it should usually be prev(HPos) (shift -1).
%   - Around the boundary (25/30), mixture is more balanced.
%
% NOTE: These probabilities are hyperparameters; tune if needed.

% 0..10 minutes (MIdx 0..2): almost always no-shift
0.97::hour_time_idx(HPos, MIdx, HPos);
0.03::hour_time_idx(HPos, MIdx, Prev) :-
    MIdx >= 0, MIdx =< 2,
    prev_idx(HPos, Prev).

% 15..20 minutes (MIdx 3..4)
0.90::hour_time_idx(HPos, MIdx, HPos);
0.10::hour_time_idx(HPos, MIdx, Prev) :-
    MIdx >= 3, MIdx =< 4,
    prev_idx(HPos, Prev).

% 25 minutes (MIdx 5): near boundary, allow more shift
0.75::hour_time_idx(HPos, 5, HPos);
0.25::hour_time_idx(HPos, 5, Prev) :-
    prev_idx(HPos, Prev).

% 30 minutes (MIdx 6): boundary, mostly shift
0.25::hour_time_idx(HPos, 6, HPos);
0.75::hour_time_idx(HPos, 6, Prev) :-
    prev_idx(HPos, Prev).

% 35 minutes (MIdx 7): strongly shift
0.10::hour_time_idx(HPos, 7, HPos);
0.90::hour_time_idx(HPos, 7, Prev) :-
    prev_idx(HPos, Prev).

% 40..55 minutes (MIdx 8..11): almost always shift
0.03::hour_time_idx(HPos, MIdx, HPos);
0.97::hour_time_idx(HPos, MIdx, Prev) :-
    MIdx >= 8, MIdx =< 11,
    prev_idx(HPos, Prev).

% Convenience: decode to actual Hour value
hour_time_from_pos_soft(HPos, MIdx, Hour) :-
    hour_time_idx(HPos, MIdx, HTimeIdx),
    idx_to_hour(HTimeIdx, Hour).

% ----------------------------------------------------------------------
% 6) Main integration: time(Image, Hour, Minute)
% ----------------------------------------------------------------------

time(X, Hour, Minute) :-
    dial(X, RIdx),
    hour_img(X, HImg),
    minute_img(X, MImg),

    rot_steps(RIdx, Steps),
    correct_idx(HImg, Steps, HPos),
    correct_idx(MImg, Steps, MIdx),

    idx_to_minute(MIdx, Minute),
    hour_time_from_pos_soft(HPos, MIdx, Hour),

    valid_time(Hour, Minute).
