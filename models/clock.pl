%:- use_module(library(nn)).

% 1. Neural Network Definition

% Takes an image (X) as input and outputs a probability distribution over 1~12
% nn(network_id, input, output_variable, domain_list)
nn(net_hour,   [X], H, [1,2,3,4,5,6,7,8,9,10,11,12]) :: hour_hand(X, H).
nn(net_minute, [X], M, [1,2,3,4,5,6,7,8,9,10,11,12]) :: minute_hand(X, M).

% 2. Clock Logic Rules (Key research component)


% Base definition of time
time(X, H, M) :-
    hour_hand(X, H),
    minute_hand(X, M),
    consistent(H, M).

% 3. Consistency Rules

% This section describes the relationship between hour and minute hands.
% Currently defined as "all combinations are possible".
% NOTE: As research progresses, this can be modified with stricter logical constraints.

consistent(H, M).
% Example: When minute hand is near 12 (00 minutes), hour hand should be exactly on an hour mark...
% More refined rules can be added later.