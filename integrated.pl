% =========================================================
% clock_integrated.pl
%   Connects:
%     - model_dial  : predicts rotation class R in {0,1,2,3}  (0°,90°,180°,270° clockwise)
%     - model_hands : predicts hand direction class in {0..11} in the *image* coordinate system
%   Then:
%     - corrects hand classes using dial rotation (derotation in logic)
%     - enforces analog-clock constraints (hour hand vs minute hand consistency)
%
% Assumptions:
%   class encoding (0..11):
%     0=12 o'clock (up), 3=3 o'clock (right), 6=6 o'clock (down), 9=9 o'clock (left)
%
% IMPORTANT:
%   Replace nn(...) names to match your Python DeepProbLog registration.
% =========================================================


% -----------------------------
% Neural predicates (DeepProbLog)
% -----------------------------
% dial: rotation class
nn(model_dial,   [Img], R,  [0,1,2,3]).
dial_rot(Img, R) :- nn(model_dial, [Img], R).

% hands: raw direction classes in image coordinates
nn(model_minute, [Img], Mr, [0,1,2,3,4,5,6,7,8,9,10,11]).
nn(model_hour,   [Img], Hr, [0,1,2,3,4,5,6,7,8,9,10,11]).

minute_raw(Img, Mr) :- nn(model_minute, [Img], Mr).
hour_raw(Img, Hr)   :- nn(model_hour,   [Img], Hr).


% -----------------------------
% Rotation correction (derotation) in logic
% -----------------------------
% R in {0,1,2,3} corresponds to {0,90,180,270} degrees clockwise.
% A 90-degree clockwise rotation shifts direction classes by +3.
% To convert raw(image-coord) -> canonical(12-at-top): Corr = (Raw - 3*R) mod 12.
rot_offset(R, Off) :- Off is (3*R) mod 12.

correct_cls(Raw, R, Corr) :-
    rot_offset(R, Off),
    Corr is (Raw - Off) mod 12.


% -----------------------------
% Domain facts (for enumerating time)
% -----------------------------
hour_val(1).  hour_val(2).  hour_val(3).  hour_val(4).  hour_val(5).  hour_val(6).
hour_val(7).  hour_val(8).  hour_val(9).  hour_val(10). hour_val(11). hour_val(12).

minute_step(0). minute_step(1). minute_step(2). minute_step(3). minute_step(4). minute_step(5).
minute_step(6). minute_step(7). minute_step(8). minute_step(9). minute_step(10). minute_step(11).


% -----------------------------
% Clock constraints
% -----------------------------
% distance on a 12-cycle (mod 12), used for tolerance
dist_mod12(A, B, D) :-
    D1 is abs(A - B),
    D2 is 12 - D1,
    ( D1 =< D2 -> D is D1 ; D is D2 ).

% tolerance in "class steps" for hour-hand consistency
% 0: strict, 1: allow +/-1 class (often helps with noise)
tolerance(1).

% expected hour-hand class for a time (H, Mstep)
% Mstep is 0..11 meaning minutes = 5*Mstep
%
% Hour-hand angle (in degrees): 30*Hbase + 0.5*minutes
% In 2.5-degree units: U = 12*Hbase + Mstep
% Convert to 30-degree class by rounding: round(U/12) = (U+6)//12
expected_hour_cls(H, Mstep, ExpCls) :-
    % Hbase: 12 -> 0, 1..11 -> 1..11
    Hbase is H mod 12,
    U is Hbase * 12 + Mstep,
    ExpCls is ((U + 6) // 12) mod 12.

% hour/minute must be consistent with analog geometry (discretized)
clock_consistent(H, Mstep, HourClsCorr) :-
    expected_hour_cls(H, Mstep, Exp),
    tolerance(T),
    dist_mod12(HourClsCorr, Exp, D),
    D =< T.


% -----------------------------
% Main predicate: combine dial + hands + constraints
% -----------------------------
% clock_time(Img, Hour, Minute)
%   Hour   : 1..12
%   Minute : 0,5,10,...,55
clock_time(Img, Hour, Minute) :-
    dial_rot(Img, R),

    minute_raw(Img, Mr),
    hour_raw(Img, Hr),

    % derotate classes using R
    correct_cls(Mr, R, Mc),    % corrected minute class
    correct_cls(Hr, R, Hc),    % corrected hour class

    % enumerate candidate time and constrain
    hour_val(Hour),
    minute_step(Mstep),
    Mc = Mstep,                % minute hand class directly determines minute step (5-min grid)
    clock_consistent(Hour, Mstep, Hc),

    Minute is Mstep * 5.


% -----------------------------
% (Optional) If your minute hand is not exactly on 5-min grid,
% comment out "Mc = Mstep" and instead allow a tolerance too.
% -----------------------------
