% Code for paper: Optimal tracking control for non-zero-sum games of linear
%             discrete-time systems via off-policy reinforcement learning.
% Method : Model based
% Programing Language : Matlab 
% Purpose : Practice and Research

clear;clc;
%% Matrix of original model
A1=[0.906488  0.0816012 -0.0005; ...
   0.074349 0.90121   -0.000708383; ...
   0         0           0.132655 ];
B1=[2;...
   1;...
    1];
D1=[1;...
   1;...
   0];
C1 = [1 1 2];
F =1;
%% 
% x(k+1)=Ax(k)+Bu(k)+Dw(k)
A = [A1 zeros(3,1);zeros(1,3) F];
B = [B1;0];
D = [D1;0];
C=[C1 0];

n=size(A,2);
m1=size(B,2);
m2=size(D,2);
p = size(C1,1);
f=(n)^2+m1^2+m2^2+m1*m2+(n)*(m1+m2)+20;
Q1_x = 20;
Q2_x = 10;
R11 = 1.25;
R12 = 3;
R21 = 4;
R22 = 1;
lamda = 0.8;
% Q1 = [C';-eye(p)']*Q1_x*[C -eye(p)];
% Q2 = [C';-eye(p)']*Q2_x*[C -eye(p)];
Q1=[C1'*Q1_x*C1 -C1'*Q1_x;...
    -Q1_x*C1 Q1_x];
Q2=[C1'*Q2_x*C1 -C1'*Q2_x;...
    -Q2_x*C1 Q2_x];
% Initial control matrix
K10=[1 0 0 0];
K20=[1 0 0 0];
K1 = {}; K1{1} = K10;
K2 = {}; K2{1} = K20;

%% Model based update
for i = 1:30
    H1  = kron(eye(4)',eye(4)')-lamda*kron((A+B*K1{i}+D*K2{i})',(A+B*K1{i}+D*K2{i})');
    H2 = (Q1+(K1{i}')*R11*K1{i}+(K2{i}')*R12*K2{i});
    H2 = H2(:);
    H3 = (Q2+(K1{i}')*R21*K1{i}+(K2{i}')*R22*K2{i});
    H3 = H3(:);
    % Solve the Riccati equation
    P1 = pinv(H1'*H1)*H1'*H2;
    P2 = pinv(H1'*H1)*H1'*H3;
    P1 = reshape(P1,4,4);
    P2 = reshape(P2,4,4);
    % Update Control Matrix
    K1{i+1}= -pinv(R11+lamda*B'*P1*B-lamda^2*B'*P1*D*pinv(R22+lamda*D'*P2*D)*D'*P2*B)*(lamda*B'*P1*A-lamda^2*B'*P1*D*pinv(R22+lamda*D'*P2*D)*D'*P2*A);
    K2{i+1}= -pinv(R22+lamda*D'*P2*D-lamda^2*D'*P2*B*pinv(R11+lamda*B'*P1*B)*B'*P1*D)*(lamda*D'*P2*A-lamda^2*D'*P2*B*pinv(R11+lamda*B'*P1*B)*B'*P1*A);
end