%% Function modified from code by Daniel Layeghi (Ref. [14] in Final Report) found under:
%% https://sites.google.com/site/daniellayeghi/daily-work-and-writing/major-project-2
function [F, J_F, q] = symbolicF(DOF)
    % Number of DOF
    n = DOF; 

    % symbolic DH parameters
    q = sym('q', [n 1], 'real'); % generalized coordinates (joint angles)
    x = sym('x', [n 1], 'real'); % generalized coordinates (joint angles)
    d = sym('d', [n 1], 'real');
    a = sym('a', [n 1], 'real'); % link offsets
    syms a1 I real
    
    % Initial conditions for the configuration of Sawyer shown in Figure 1.
    d0 = [.317 .1925 .400 .1685 .400 .1363 .13375];
    x0 = [-pi/2 -pi/2 -pi/2 -pi/2 -pi/2 -pi/2 0];
    a0 = [.081 0 0 0 0 0 0]; 

    % Cell array of your homogeneous transformations; each Ti{i} is a 4x4 symbolic transform matrix
    Ti = cell(n + 1,1);
    Ti{1} = {[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]};
    for i = 2:n+1
        Ti{i} =  Ti{i-1} * ([cos(q(i-1)) -sin(q(i-1)) 0 0; sin(q(i-1)) cos(q(i-1)) 0 0;...
                             0 0 1 0; 0 0 0 1] *[1 0 0 0; 0 1 0 0; 0 0 1 d(i-1); 0 0 0 1]*[1 0 0 a(i-1);...
                             0 1 0 0; 0 0 1 0; 0 0 0 1]*[1 0 0 0; 0 cos(x(i-1)) -sin(x(i-1)) 0;...
                             0 sin(x(i-1)) cos(x(i-1)) 0; 0 0 0 1]);
    end
    Ti = Ti(2:n + 1); % Exclude the identity at Ti{1}
    
    % Substitute the numerical DH parameters for the Sawyer arm
    for i = 1:n
        Titemp = subs(Ti{i}, x, x0');
        Titemp = subs(Titemp, a, a0');
        Titemp = subs(Titemp, d, d0');
        Ti{i} = Titemp;
    end
    
    % Transform matrix from EEF to base
    finalTrans = Ti{end}; 
    
    % Endpoint coordinate map in the base frame
    F = finalTrans*[0; 0; 0; 1];
    F = F(1:3);
    
    % Jacobian of the endpoint coordinate map with respect to q
    J_F = simplify(jacobian(F, q));
    J_F = J_F(:, 1:n - 1); % Jacobian of the endpoint position is not a function of q7
end