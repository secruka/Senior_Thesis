% ============================================
% clock_integrated.pl
% DeepProbLog integrated program:
%   - net_hour   : 12-class (1..12)
%   - net_minute : 12-class (1..12)  where 12 means "00 minutes"
%   - net_rot    : 12-class (0,30,...,330) degrees (clockwise)
%
% Goal predicate (for training/eval):
%   time(X, H, M).
% ============================================

% ---------- Neural predicates ----------
nn(net_hour,   X, Hraw, [1,2,3,4,5,6,7,8,9,10,11,12]).
nn(net_minute, X, Mraw, [1,2,3,4,5,6,7,8,9,10,11,12]).
nn(net_rot,    X, Rdeg, [0,30,60,90,120,150,180,210,240,270,300,330]).

% Wrapper predicates for neural networks
net_hour(X, H) :- nn(net_hour, X, H, _).
net_minute(X, M) :- nn(net_minute, X, M, _).
net_rot(X, R) :- nn(net_rot, X, R, _).

% ---------- Utilities ----------
% Safe modulo 360 that works for negatives as well
mod360(A, B) :-
    B is ((A mod 360) + 360) mod 360.

% ---------- Label <-> degree maps ----------
% minute index mapping:
%   12 -> 0deg (00min)
%   1  -> 30deg (05min)
%   2  -> 60deg (10min)
%   ...
%   11 -> 330deg (55min)
minute_deg(12, 0).
minute_deg(1,  30).
minute_deg(2,  60).
minute_deg(3,  90).
minute_deg(4,  120).
minute_deg(5,  150).
minute_deg(6,  180).
minute_deg(7,  210).
minute_deg(8,  240).
minute_deg(9,  270).
minute_deg(10, 300).
minute_deg(11, 330).

% hour mapping:
%   12 -> 0deg
%   1  -> 30deg
%   ...
%   11 -> 330deg
hour_deg(12, 0).
hour_deg(1,  30).
hour_deg(2,  60).
hour_deg(3,  90).
hour_deg(4,  120).
hour_deg(5,  150).
hour_deg(6,  180).
hour_deg(7,  210).
hour_deg(8,  240).
hour_deg(9,  270).
hour_deg(10, 300).
hour_deg(11, 330).

% Inverse maps are already covered by minute_deg/2 and hour_deg/2
% because the mapping is one-to-one for these discrete bins.

% ---------- Rotation correction ----------
% Assumption:
%   - CNNs output "raw" angles as seen in the image.
%   - net_rot outputs the dial rotation offset (clockwise degrees).
%   - Corrected angle = raw - rotation (mod 360)
%
% After correcting degrees, map back to the discrete label.

% ---------- Final predicate ----------
% Keep the signature compatible with your existing DeepProbLog queries:
%   time(X, H, M).
time(X, H, M) :-
    net_hour(X, H),
    net_minute(X, M).

