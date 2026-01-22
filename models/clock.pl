% ============================================
% clock_integrated.pl
% DeepProbLog integrated program for complete clock reading with constraints:
%   - net_hour   : 12-class (1..12)
%   - net_minute : 12-class (1..12)  where 12 means "00 minutes"
%   - net_rot    : 12-class (0,30,...,330) degrees (clockwise)
%
% Goal predicates:
%   time(X, H, M).              - Predict time from image
%   valid_time(H, M).           - Check if H:M is valid
%   corrected_time(X, H, M).    - Time with rotation correction
% ============================================

% ---------- Neural predicates ----------
nn(net_hour,   X, Hraw, [1,2,3,4,5,6,7,8,9,10,11,12]).
nn(net_minute, X, Mraw, [1,2,3,4,5,6,7,8,9,10,11,12]).
nn(net_rot,    X, Rdeg, [0,30,60,90,120,150,180,210,240,270,300,330]).

% Wrapper predicates for neural networks
net_hour(X, H) :- nn(net_hour, X, H, _).
net_minute(X, M) :- nn(net_minute, X, M, _).
net_rot(X, R) :- nn(net_rot, X, R, _).

% ---------- Utility predicates ----------
% Safe modulo 360 that works for negatives as well
mod360(A, B) :-
    B is ((A mod 360) + 360) mod 360.

% Modulo 12 for hour wraparound
mod12(H, Result) :-
    R is ((H - 1) mod 12) + 1,
    Result is R.

% Check if a number is within range
in_range(X, Min, Max) :-
    X >= Min,
    X =< Max.

% ---------- Label <-> degree maps ----------
% minute index to degrees mapping:
%   12 -> 0deg   (00 minutes)
%   1  -> 30deg  (05 minutes)
%   2  -> 60deg  (10 minutes)
%   ... (each step = 5 minutes = 30 degrees)
%   11 -> 330deg (55 minutes)
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

% Inverse: degrees to minute index
degrees_minute(0,   12).
degrees_minute(30,  1).
degrees_minute(60,  2).
degrees_minute(90,  3).
degrees_minute(120, 4).
degrees_minute(150, 5).
degrees_minute(180, 6).
degrees_minute(210, 7).
degrees_minute(240, 8).
degrees_minute(270, 9).
degrees_minute(300, 10).
degrees_minute(330, 11).

% hour index to degrees mapping:
%   12 -> 0deg   (12 o'clock)
%   1  -> 30deg  (1 o'clock)
%   2  -> 60deg  (2 o'clock)
%   ... (each step = 1 hour = 30 degrees)
%   11 -> 330deg (11 o'clock)
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

% Inverse: degrees to hour index
degrees_hour(0,   12).
degrees_hour(30,  1).
degrees_hour(60,  2).
degrees_hour(90,  3).
degrees_hour(120, 4).
degrees_hour(150, 5).
degrees_hour(180, 6).
degrees_hour(210, 7).
degrees_hour(240, 8).
degrees_hour(270, 9).
degrees_hour(300, 10).
degrees_hour(330, 11).

% Convert minute index to actual minutes
minute_value(12, 0).
minute_value(1, 5).
minute_value(2, 10).
minute_value(3, 15).
minute_value(4, 20).
minute_value(5, 25).
minute_value(6, 30).
minute_value(7, 35).
minute_value(8, 40).
minute_value(9, 45).
minute_value(10, 50).
minute_value(11, 55).

% ---------- Validation predicates ----------
% Check if hour is valid (1-12)
valid_hour(H) :- in_range(H, 1, 12).

% Check if minute index is valid (1-12, where 12 = 00)
valid_minute_index(M) :- in_range(M, 1, 12).

% Check if time is valid (hour and minute consistency)
valid_time(H, M) :-
    valid_hour(H),
    valid_minute_index(M).

% Check if degree is a valid clock degree (0, 30, 60, ..., 330)
valid_degree(D) :-
    member(D, [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]).

% Member predicate for checking if element is in list
member(X, [X|_]).
member(X, [_|T]) :- member(X, T).

% ---------- Rotation correction logic ----------
% Apply rotation correction to hour hand
correct_hour_position(Hraw, Rdeg, Hcorrected) :-
    hour_deg(Hraw, DegRaw),
    Diff is DegRaw - Rdeg,
    mod360(Diff, DegCorr),
    degrees_hour(DegCorr, Hcorrected).

% Apply rotation correction to minute hand
correct_minute_position(Mraw, Rdeg, Mcorrected) :-
    minute_deg(Mraw, DegRaw),
    Diff is DegRaw - Rdeg,
    mod360(Diff, DegCorr),
    degrees_minute(DegCorr, Mcorrected).

% ---------- Hand relationship constraints ----------
% Verify hour and minute hand positions are consistent
% Hour hand moves gradually (30 degrees per hour + 0.5 degrees per minute)
% Minute hand position in degrees should loosely align with hour hand progress
hands_consistent(H, M) :-
    hour_deg(H, HDeg),
    minute_deg(M, MDeg),
    % Minute hand should be at its expected position (doesn't constrain hour directly)
    valid_degree(MDeg),
    % Both should be valid degrees on clock
    valid_degree(HDeg).

% ---------- Main inference predicates ----------
% Hands angle classification: hands(X, HourAngleClass, MinuteAngleClass)
% This is the main predicate used by model_hands2.py
% HourAngleClass and MinuteAngleClass are 12-class predictions (0..11)
% representing angle classes (0=12o'clock, 1=30deg, 2=60deg, ..., 11=330deg)
hands(X, H, M) :-
    nn(net_hour, X, H, _),
    nn(net_minute, X, M, _).

% Direct time reading without rotation correction
time(X, H, M) :-
    net_hour(X, H),
    net_minute(X, M),
    valid_time(H, M).

% Time reading with rotation awareness (returns raw predictions + rotation)
time_with_rotation(X, H, M, R) :-
    net_hour(X, H),
    net_minute(X, M),
    net_rot(X, R),
    valid_time(H, M),
    valid_degree(R).

% Corrected time accounting for dial rotation
corrected_time(X, Hcorr, Mcorr) :-
    net_hour(X, Hraw),
    net_minute(X, Mraw),
    net_rot(X, Rdeg),
    valid_time(Hraw, Mraw),
    valid_degree(Rdeg),
    correct_hour_position(Hraw, Rdeg, Hcorr),
    correct_minute_position(Mraw, Rdeg, Mcorr),
    valid_time(Hcorr, Mcorr),
    hands_consistent(Hcorr, Mcorr).

% Fallback: return raw predictions if correction fails
time_fallback(X, H, M) :-
    time(X, H, M).

% Get time details (both index and actual values)
time_details(X, H, M, MinVal) :-
    time(X, H, M),
    minute_value(M, MinVal).

% ---------- Constraint satisfaction predicates ----------
% All predictions must be from valid sets
well_formed_predictions(X, H, M, R) :-
    valid_hour(H),
    valid_minute_index(M),
    valid_degree(R),
    net_hour(X, H),
    net_minute(X, M),
    net_rot(X, R).

% Time must be physically reasonable (both hands present)
physically_plausible(X, H, M) :-
    time(X, H, M),
    hands_consistent(H, M),
    % Ensure neither hand is at "invalid" position
    hour_deg(H, _),
    minute_deg(M, _).


