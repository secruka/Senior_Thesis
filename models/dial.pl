% ============================================
% dial.pl
% DeepProbLog program for dial rotation classification
% ============================================

% Neural network predicate for rotation classification
% Outputs one of [0,30,60,90,120,150,180,210,240,270,300,330] degrees
nn(net_rot, X, R, [0,30,60,90,120,150,180,210,240,270,300,330]).

% Wrapper predicate that calls the neural network
net_rot(X, R) :- nn(net_rot, X, R, _).

% Main query predicate: rot(image, rotation_degrees)
rot(X, R) :- net_rot(X, R).
