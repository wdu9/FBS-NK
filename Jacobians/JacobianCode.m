
%parameters

N_ss = 1.19
tau =   0.16563445378151262
w_ss = 1/1.012
Z = 1

%%
H_U = zeros(600,600)

for n= 1:3:600
    H_U(n,1) =2
end

H_ut =zeros(3,600)



H_ut(1,1:200) = CJAC(:,1)


H_ut(1,201:400) = CJACW(:,1)

H_ut(1,401:600) = CJACN(:,1)

H_ut(1,201) = CJACW(1,1) - tau*N_ss

H_ut(1,401) = CJACN(1,1) - Z - w_ss*tau






