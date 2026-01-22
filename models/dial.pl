% ============================================
% dial.pl
% DeepProbLog program for dial rotation detection and correction
% 
% Handles clock dial rotation classification:
%   - net_rot: 12-class (0,30,60,...,330) degrees (clockwise)
%
% Goal predicates:
%   rot(X, R).              - Predict dial rotation
%   valid_rotation(R).      - Check if rotation is valid
%   rotation_offset(X, R).  - Get rotation with validation
% ============================================

% ---------- Neural predicates ----------
% Neural network for rotation classification
% Outputs one of 4 classes (0=0°, 1=90°, 2=180°, 3=270°)
nn(net_dial, X, R, [0,1,2,3]).

% Wrapper predicate that calls the neural network
net_dial(X, R) :- nn(net_dial, X, R, _).

% Original rotation network for compatibility
% Outputs one of [0,30,60,90,120,150,180,210,240,270,300,330] degrees
nn(net_rot, X, R, [0,30,60,90,120,150,180,210,240,270,300,330]).

% Wrapper predicate that calls the neural network
net_rot(X, R) :- nn(net_rot, X, R, _).

% Main dial predicate: maps 4-class rotation (0..3) to dial classification
dial(X, RotCls) :-
    nn(net_dial, X, RotCls, _).

% ---------- Utility predicates ----------
% List of valid rotation degrees (12-class, 30 degrees apart)
valid_rotation_degree(0).
valid_rotation_degree(30).
valid_rotation_degree(60).
valid_rotation_degree(90).
valid_rotation_degree(120).
valid_rotation_degree(150).
valid_rotation_degree(180).
valid_rotation_degree(210).
valid_rotation_degree(240).
valid_rotation_degree(270).
valid_rotation_degree(300).
valid_rotation_degree(330).

% Member predicate for list checking
member(X, [X|_]).
member(X, [_|T]) :- member(X, T).

% Check if rotation degree is valid
valid_rotation(R) :-
    valid_rotation_degree(R).

% Get list of all valid rotations
all_rotations([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]).

% ---------- Rotation normalization ----------
% Normalize angle to 0-359 range
normalize_angle(Angle, Normalized) :-
    Normalized is ((Angle mod 360) + 360) mod 360.

% Check if two angles are equivalent (within tolerance of 0.5 degrees)
angles_equivalent(A1, A2) :-
    normalize_angle(A1, N1),
    normalize_angle(A2, N2),
    Diff is abs(N1 - N2),
    (Diff =< 0.5 ; Diff >= 359.5).

% ---------- Rotation inference predicates ----------
% Main predicate: predict dial rotation from image
rot(X, R) :- 
    net_rot(X, R),
    valid_rotation(R).

% Get rotation with additional validation
rotation_offset(X, R) :-
    net_rot(X, R),
    valid_rotation(R).

% Rotation with confidence check (can be extended with confidence scores)
confident_rotation(X, R) :-
    net_rot(X, R),
    valid_rotation(R).

% ---------- Rotation correction rules ----------
% Correct an angle by removing detected rotation
correct_angle_by_rotation(OriginalAngle, RotationOffset, CorrectedAngle) :-
    valid_rotation(RotationOffset),
    CorrectedAngle is OriginalAngle - RotationOffset,
    normalize_angle(CorrectedAngle, CorrectedAngle).

% Inverse: add rotation back to an angle
apply_rotation(OriginalAngle, RotationOffset, RotatedAngle) :-
    valid_rotation(RotationOffset),
    RotatedAngle is OriginalAngle + RotationOffset,
    normalize_angle(RotatedAngle, RotatedAngle).

% ---------- Rotation classification refinement ----------
% Quantize continuous angle to nearest valid rotation (for post-processing)
quantize_to_valid_rotation(ContinuousAngle, QuantizedRotation) :-
    valid_rotation_degree(QuantizedRotation),
    Diff is abs(ContinuousAngle - QuantizedRotation),
    \+ (valid_rotation_degree(Other),
        Other \= QuantizedRotation,
        OtherDiff is abs(ContinuousAngle - Other),
        OtherDiff < Diff).

% Get closest valid rotation to an arbitrary angle
closest_valid_rotation(Angle, ClosestRotation) :-
    normalize_angle(Angle, NormAngle),
    quantize_to_valid_rotation(NormAngle, ClosestRotation).

% ---------- Consistency checks ----------
% Verify rotation makes physical sense (no "flipped" dial)
physically_plausible_rotation(R) :-
    valid_rotation(R),
    % Rotation should be in [0, 330], not extreme values
    R >= 0,
    R =< 330.

% Check if two rotations are reasonably close (within one class = 30 degrees)
rotations_close(R1, R2) :-
    valid_rotation(R1),
    valid_rotation(R2),
    Diff is abs(R1 - R2),
    (Diff =< 30 ; Diff >= 330).

% ---------- Rotation tracking (for temporal consistency) ----------
% Check if current rotation is reasonable given previous rotation
% (for video sequences or batch processing)
reasonable_rotation_change(PrevRotation, CurrentRotation) :-
    valid_rotation(PrevRotation),
    valid_rotation(CurrentRotation),
    % Allow change of at most 2 classes (60 degrees)
    Diff is abs(PrevRotation - CurrentRotation),
    (Diff =< 60 ; Diff >= 300).

% ---------- Ensemble/aggregation predicates ----------
% Average two valid rotations (normalized)
average_rotations(R1, R2, Average) :-
    valid_rotation(R1),
    valid_rotation(R2),
    Sum is R1 + R2,
    RawAvg is Sum / 2,
    normalize_angle(RawAvg, Average).

% Select most likely rotation from predictions
most_likely_rotation(Rotations, MostLikely) :-
    member(MostLikely, Rotations),
    valid_rotation(MostLikely),
    \+ (member(Other, Rotations),
        Other \= MostLikely,
        valid_rotation(Other)).

