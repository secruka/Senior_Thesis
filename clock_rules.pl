% clock_rules.pl
% 角度は「12時方向=0度、時計回りに増加」とする。
% 浮動小数を避けるため、角度は「0.1度単位（tenths）」で扱う。
% 例: 90.0度 -> 900, 127.5度 -> 1275

% ---------- 基本変換（時刻 -> 角度） ----------

% 分針角度(0.1度): minute_angle10(M, AMin10)
minute_angle10(M, AMin10) :-
    must_be(integer, M),
    M >= 0, M =< 59,
    AMin10 is (M * 60) mod 3600.

% 短針角度(0.1度): hour_angle10(H, M, AHour10)
% 30度/時 = 300(0.1度), 0.5度/分 = 5(0.1度)
hour_angle10(H, M, AHour10) :-
    must_be(integer, H), must_be(integer, M),
    H >= 0, H =< 11, M >= 0, M =< 59,
    AHour10 is ((H * 300) + (M * 5)) mod 3600.

% まとめて出す
angles_from_time(H, M, AMin10, AHour10) :-
    minute_angle10(M, AMin10),
    hour_angle10(H, M, AHour10).

% ---------- 角度の近さ判定（許容誤差 tol を使う） ----------

% circular distance in [0,1800] tenths
circ_dist10(A, B, D) :-
    A0 is A mod 3600, B0 is B mod 3600,
    Diff is abs(A0 - B0),
    D is min(Diff, 3600 - Diff).

close10(A, B, Tol10) :-
    circ_dist10(A, B, D),
    D =< Tol10.

% ---------- 角度 -> 時刻（探索で復元） ----------
% time_from_angles(AMin10, AHour10, TolMin10, TolHour10, H, M)
% 例: TolMin10=30 は ±3.0度 の許容（分針用）
%     TolHour10=40 は ±4.0度 の許容（短針用）
time_from_angles(AMin10, AHour10, TolMin10, TolHour10, H, M) :-
    between(0, 59, M),
    minute_angle10(M, AMinExp),
    close10(AMin10, AMinExp, TolMin10),
    between(0, 11, H),
    hour_angle10(H, M, AHourExp),
    close10(AHour10, AHourExp, TolHour10).

% ---------- 盤面回転オフセットがある場合（任意） ----------
% rot は 0,90,180,270度を想定（0.1度単位で0,900,1800,2700）
apply_rot10(Rot10, AIn10, AOut10) :-
    AOut10 is (AIn10 - Rot10) mod 3600.

% 角度観測が「回転した盤面基準」なら、補正してから time_from_angles を呼ぶ
time_from_angles_with_rot(AMinObs10, AHourObs10, Rot10, TolMin10, TolHour10, H, M) :-
    apply_rot10(Rot10, AMinObs10, AMin10),
    apply_rot10(Rot10, AHourObs10, AHour10),
    time_from_angles(AMin10, AHour10, TolMin10, TolHour10, H, M).
