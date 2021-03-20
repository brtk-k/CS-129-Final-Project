% Code that generates a matrix to transform from the EEF to the base frame
% Adapted from D. Layeghi Code, also titled transforms.m, found at
% https://sites.google.com/site/daniellayeghi/daily-work-and-writing/major-project-2

n = 7; % DOF

% DH parameters
q = sym('q', [n 1], 'real'); % generalized coordinates (joint angles)
x = sym('x', [n 1], 'real'); % generalized coordinates (joint angles)
d = sym('d', [n 1], 'real');
a = sym('a', [n 1], 'real'); % link offsets
syms a1 I real


% initial conditions for the configuration of Sawyer shown in Figure 1.

q0 = [0 3*pi/2 0 pi 0 pi 3*pi/2];
d0 = [.317 .1925 .400 .1685 .400 .1363 .13375];
x0 = [-pi/2 -pi/2 -pi/2 -pi/2 -pi/2 -pi/2 0];
a0 = [.081 0 0 0 0 0 0]; 
I0 = 1;

% cell array of your homogeneous transformations; each Ti{i} is a 4x4 symbolic transform matrix
Ti = cell(n+1,1);
Ti(1) = {[1 0 0 0;0 1 0 0; 0 0 1 0; 0 0 0 1]};
% Ti{i-1} *
for i = 2:n+1
    Ti{i} =  Ti{i-1} * ([cos(q(i-1)) -sin(q(i-1)) 0 0; sin(q(i-1)) cos(q(i-1)) 0 0; 0 0 1 0; 0 0 0 1] *[1 0 0 0; 0 1 0 0; 0 0 1 d(i-1); 0 0 0 1]*[1 0 0 a(i-1); 0 1 0 0; 0 0 1 0; 0 0 0 1]*[1 0 0 0; 0 cos(x(i-1)) -sin(x(i-1)) 0 ; 0 sin(x(i-1)) cos(x(i-1)) 0; 0 0 0 1]);
end
Ti = Ti(2:8 , 1);
Tisub = [];
%Ti = {Ti(2:8 , 1)};
for i = 1:n
    Titemp = subs(Ti{i,1}, x, x0');
    Titemp = subs(Titemp, a, a0');
    Titemp = subs(Titemp, d, d0');
    Tisub = [Tisub; Titemp];
end
Ti_c = cell (n,1);
for k = 1:n 
    Ti_c{k} = Tisub(4*k-3:4*k,1:4); 
end 
Ti = cell(n,1);
Ti = Ti_c;
finalTrans = Ti{end}; %Transform matrix from EEF to base

%% Testing
% q00 = [2.1591796875E-03	5.4573828125E-01	-3E-03	-1E-03	3.1E-02	8.8E-02	-2.728E+00];
% % Coordinates 2-8 are joint rotations
% attempt = subs(finalTrans, q, q00');
% Euclidcoord = attempt * [0;0;0;1];
% X = double(Euclidcoord(1))
% Y = double(Euclidcoord(2))
% Z = double(Euclidcoord(3))