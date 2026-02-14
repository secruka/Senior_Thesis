%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% models/clock_latent.pl
%
% DeepProbLog program for latent-variable clock recognition.
%
% Supervision: time(Image, Hour, Minute) ONLY.
% The decomposition into dial rotation (R), hour-hand angle (H),
% and minute-hand angle (M) is fully latent — DeepProbLog marginalises
% over all valid (R, H, M) groundings that satisfy the time constraint.
%
% Neural predicates (all outputs are latent — no direct labels):
%   net_rot    : dial rotation classifier       (4 classes: 0/90/180/270)
%   net_hour   : hour-hand angle classifier     (12 classes, 30-deg bins, image coords)
%   net_minute : minute-hand angle classifier   (12 classes, 30-deg bins, image coords)
%
% Key constraint (hour-minute coupling):
%   When the minute hand is in the first half (0-25 min), the hour hand
%   points at the current hour tick. When it is in the second half
%   (30-55 min), the hour hand has moved past the tick, so the actual
%   hour = tick - 1. This coupling means Hour cannot be decoded from
%   the hour-hand angle alone — it depends on the minute-hand position,
%   making DeepProbLog's marginalisation essential for learning.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- neural predicates (output softmax probabilities) ---
nn(net_rot,    [X], R, [0,1,2,3]) :: dial(X, R).
nn(net_hour,   [X], H, [0,1,2,3,4,5,6,7,8,9,10,11]) :: hour_raw(X, H).
nn(net_minute, [X], M, [0,1,2,3,4,5,6,7,8,9,10,11]) :: minute_raw(X, M).

% --- rotation -> 12-step offset ---
rot_steps(0, 0).   % 0 deg
rot_steps(1, 3).   % 90 deg CW
rot_steps(2, 6).   % 180 deg
rot_steps(3, 9).   % 270 deg CW

% --- modular correction: image coords -> canonical coords ---
correct_idx(ImageIdx, Steps, CanonIdx) :-
    CanonIdx is (ImageIdx - Steps + 120) mod 12.

% --- previous index on 12-cycle ---
prev_idx(I, P) :-
    P is (I - 1 + 12) mod 12.

% --- canonical index -> display values ---
idx_to_hour(0, 12).
idx_to_hour(I, I) :- I >= 1, I =< 11.

idx_to_minute(MIdx, Min) :-
    Min is MIdx * 5.

% --- clock constraint: hour depends on minute position ---
% When minute hand is in first half (0-25 min, MIdx 0..5):
%   hour hand points at the current hour tick
hour_from_pos(HPos, MIdx, Hour) :-
    MIdx < 6,
    idx_to_hour(HPos, Hour).

% When minute hand is in second half (30-55 min, MIdx 6..11):
%   hour hand has moved past the hour tick, so actual hour = tick - 1
hour_from_pos(HPos, MIdx, Hour) :-
    MIdx >= 6,
    prev_idx(HPos, HPrev),
    idx_to_hour(HPrev, Hour).

% --- time validity check ---
valid_time(H, M) :-
    H >= 1, H =< 12,
    M >= 0, M =< 55,
    0 is M mod 5.

% --- sole query predicate: time(Image, Hour, Minute) ---
% R, H, M are all latent — marginalised by DeepProbLog.
time(X, Hour, Minute) :-
    dial(X, RIdx),
    hour_raw(X, HImg),
    minute_raw(X, MImg),
    rot_steps(RIdx, Steps),
    correct_idx(HImg, Steps, HPos),
    correct_idx(MImg, Steps, MPos),
    idx_to_minute(MPos, Minute),
    hour_from_pos(HPos, MPos, Hour),
    valid_time(Hour, Minute).
