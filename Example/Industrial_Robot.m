clear
close all
clc
load('DataSet/forward_identification_without_raw_data.mat') %loading the data set
 u_std = std(u_train,0,2); 
 y_std = std(y_train,0,2);
  u_mean = mean(u_train,2); 
  y_mean = mean(y_train,2);
 devia=y_std;
for k_ax = 1:size(y_train,1)
    y_test(k_ax,:) = (y_test(k_ax,:)-y_mean(k_ax)) / y_std(k_ax);
    y_train(k_ax,:) = (y_train(k_ax,:)-y_mean(k_ax)) / y_std(k_ax);
    u_train(k_ax,:) = (u_train(k_ax,:)-u_mean(k_ax)) / u_std(k_ax);
    u_test(k_ax,:) = (u_test(k_ax,:)-u_mean(k_ax)) / u_std(k_ax);
end

% How many trajectories we want to obtain
NeuronNum_switch=600;
NeuronNum_single=2000;
% Sampling data from working zone
u=0.2;
delta=0.001;
tau=0.01;
TF='sig';
%duration = size(y_test,2);
duration =3636;
SystemState=24;
ControlState=24;
e=5e-6;


tol = 2;
maximum_entropy=1500;

% Initial Input Set
%durationReach=20;

%% Load data set
%load(['DataSet/forward_identification_without_raw_data.mat'])
xs = zeros(size(u_train,2)-SystemState+1,SystemState*6+6*ControlState);
t = zeros(size(u_train,2)-SystemState+1,6);
for i = 1:size(u_train,2)-SystemState
   for j = 1:SystemState
    xs(i,1+(j-1)*6:j*6)=y_train(:,i+(j-1))';
   end
   for k= 1:ControlState
       xs(i,j*6+1+(k-1)*6:j*6+k*6)=u_train(:,i+SystemState-1-ControlState+k)';
   end
   t(i,:)=y_train(:,i+SystemState)';  
end
for i = 1:size(y_test,1)
    subplot(size(y_test,1),1,i)
    plot(time_test,y_test(i,:))
    xlabel('time(s)')
    ylabel('position')
    hold on
end



% 
% delta_y = zeros(6,size(time_train,2)-SystemState+1);
% 
% for i = 1:size(time_train,2)-SystemState+1
%    delta_y(:,i)=abs(y_train(:,i+1)-y_train(:,i));
% end
% MaxDelta_y=zeros(size(y_train,1),1);
% for i =1:6
%  MaxDelta_y(i,1)=max(delta_y(i,:));
% end
%xs=u_train';
%t=y_train';

[Pn_train,inputps] = mapminmax(xs');
%Pn_test = mapminmax('apply',P_test,inputps);
[Tn_train,outputps] = mapminmax(t');
%% Princeple Component Analysis
% [coeff,scoreeeTrain,~,~,explained,mu] = pca(xs(:,1:12));
% idx=find(cumsum(explained)>90,1);      
% utest = (xs(:,1:12)-mu)*coeff(:,1:idx);
% bounderies = zeros(idx,2);
[coeff,scoreeeTrain,~,~,explained,mu] = pca(Pn_train(1:6*SystemState,:)');
idx=find(cumsum(explained)>90,1);      
utest = (Pn_train(1:6*SystemState,:)'-mu)*coeff(:,1:idx);
bounderies = zeros(idx,2);

for i= 1:idx
    bounderies(i,1)=min(utest(:,i))-2;
    bounderies(i,2)=max(utest(:,i))+2;
end
lowerbound=bounderies(:,1);
upperbound= bounderies(:,2);
%init_interval{1}=[lowerbound',upperbound'];
init_interval{1}=bounderies;
Dimension=size(bounderies,1);

%% Data-driven Partitioning
l1=tic;


P=partitions(init_interval,Pn_train(1:6*SystemState,:)',Tn_train');
intervals=ME(P,tol,maximum_entropy,Dimension,mu',coeff(:,1:idx)');
%intervals=ME(P,tol,maximum_entropy,Dimension,mu',coeff(:,1:idx)');
P.intervals=intervals;
P.input=Pn_train';

%% Initialized ELM and Merge Partitions
ELMs1=ELM.GenerateELM(size(xs,2),NeuronNum_switch,TF,size(t,2));
toc(l1)
[P1,ELMs]=MergePatitions(P,ELMs1,e,mu',coeff(:,1:idx)',6*SystemState);
size(P.intervals,2)
size(P1.intervals,2)
mse_switch = 0;
min_mse_switch=1;
for i = 1:size(ELMs,2)
    if (mse_switch<ELMs(i).trainingError)
            mse_switch = ELMs(i).trainingError;
            k=i;
    end
    if (min_mse_switch>ELMs(i).trainingError)
            min_mse_switch = ELMs(i).trainingError;
    end
end

%  for i = 1:size(ELMs,2)
%     ELMs(i).trainingError
%      i         
%  end

% Plot intervals
% figure
% partitions.intervalplot(P.intervals,'empty','blue')
% partitions.intervalplot(P1.intervals,'full','red')
% title('Invariantspace using Bisection method')

%% Train a Complex Neural Network Model as referance
l2=tic;
ELMs1=ELM.GenerateELM(size(xs,2),NeuronNum_single,TF,size(t,2));
%ELMs1=trainELMLipridge(ELMs1,xs',t');
ELMs1=trainELM(ELMs1,Pn_train,Tn_train);
toc(l2)
%% Verfify whether can approximate the model well
output_single=zeros(6,size(y_test,2)-SystemState);
output_switch=output_single;
segmentIndex=P1.intervals;
inputspace1=P1.intervals;
%   xs(i,:)=[y_train(:,i)',y_train(:,i+1)',u_train(:,i+1)'];
%   t(i,:)=[y_train(:,i+2)'];  

for i =1:size(time_test,2)-SystemState
    for j= 1:SystemState
        yinput(1+(j-1)*6:j*6,1)=y_test(:,i+j-1);
    end
    for k= 1:ControlState
    %   xs(i,j*6+1+(k-1)*6:j*6+k*6)=u_train(:,i+SystemState-k)';
       yinput(SystemState*6+1+(k-1)*6:SystemState*6+k*6,:)=u_test(:,i+SystemState-1-ControlState+k);
    end
    
    IN=mapminmax('apply',yinput,inputps);
     output_single(:,i) = ELMpredict(ELMs1,IN);
  
   for k = 1:size(segmentIndex,2)
              if(partitions.ifin(coeff(:,1:idx)'*(IN(1:SystemState*6,:)-mu'),segmentIndex{k},Dimension)==1)
                       output_switch(:,i)= ELMpredict(ELMs(k),IN);
              end
    end
end

%BPs=newff(xs',t',[18 200 200 6],{'purelin' 'tansig' 'tansig' 'purelin'});
%BPs=train(BPs,xs',t');
%% Analysis from the stastical point of view
 % 1.Plot Figures
figure
ON= mapminmax('reverse',output_single(:,1:end),outputps);
for k_ax = 1:6
  ON(k_ax,:) = ON(k_ax,:) * y_std(k_ax) + y_mean(k_ax);
  y_test(k_ax,:) = y_test(k_ax,:) * y_std(k_ax) + y_mean(k_ax);
end
t = tiledlayout(size(y_test,1),1,'TileSpacing','Compact');
t.TileSpacing = 'compact';
t.Padding = 'compact';
for i = 1:size(y_test,1)
nexttile
   % tsubplot(size(y_test,1),1,'tight')
    % set(gca,'position',[0.08,0.75,0.89,0.23])
    plot(time_test(SystemState+1:end),y_test(i,SystemState+1:end))
    hold on 
    plot(time_test(SystemState+1:end),ON(i,1:end))
%    xlabel('time(s)')
%    ylabel('position')
%    hold on
 end
 %title('single neural network modeling')

ON_S= mapminmax('reverse',output_switch(:,1:end),outputps);
for k_ax = 1:6
  ON_S(k_ax,:) = ON_S(k_ax,:) * y_std(k_ax) + y_mean(k_ax);
end
figure
for i = 1:size(y_test,1)
    subplot(size(y_test,1),1,i)
    plot(time_test(SystemState+1:end),y_test(i,SystemState+1:end))
    hold on 
    plot(time_test(SystemState+1:end),ON_S(i,1:end))
   % xlabel('time(s)')
    %ylabel('position')
    hold on
end
title('switching neural network modeling')

ON= mapminmax('reverse',output_single(:,1:end),outputps);
for k_ax = 1:6
  ON(k_ax,:) = ON(k_ax,:) * y_std(k_ax) + y_mean(k_ax);
end
t = tiledlayout(size(y_test,1),1,'TileSpacing','Compact');
t.TileSpacing = 'compact';
t.Padding = 'compact';
for i = 1:size(y_test,1)
    
nexttile
   % tsubplot(size(y_test,1),1,'tight')
    % set(gca,'position',[0.08,0.75,0.89,0.23])
    plot(time_test(SystemState+1:end),y_test(i,SystemState+1:end))
    hold on 
    plot(time_test(SystemState+1:end),ON_S(i,1:end))
%    xlabel('time(s)')
%    ylabel('position')
%    hold on
 end
 %title('single neural network modeling')


%% Simulate the Respones
Traj=zeros(size(y_test,1),duration);
for j = 1:SystemState
    Traj(:,j)=[y_test(:,j)];
end
Traj_single=Traj;
Traj_switch=Traj;
%inputN(:,1:3)= Traj

for i =SystemState+1:duration
    for j = 1:SystemState
        inputn(1+(j-1)*6:j*6,:)= Traj_single(:,i-SystemState+j-1);
    end
    for k= 1:ControlState
    %   xs(i,j*6+1+(k-1)*6:j*6+k*6)=u_train(:,i+SystemState-k)';
       inputn(SystemState*6+1+(k-1)*6:SystemState*6+k*6,:)=u_test(:,i-1-ControlState+k);
    end
    %inputn(SystemState*6+1:SystemState*6+6,:)=u_test(:,i-1);
    InputN=mapminmax('apply',inputn,inputps);
    
    Traj_singleN=ELMpredict(ELMs1,InputN);
    Traj_single(:,i)=mapminmax('reverse',Traj_singleN,outputps);
    for j = 1:SystemState
        inputns(1+(j-1)*6:j*6,:)= Traj_switch(:,i-SystemState+j-1);
    end
    for k= 1:ControlState
    %   xs(i,j*6+1+(k-1)*6:j*6+k*6)=u_train(:,i+SystemState-k)';
       inputns(SystemState*6+1+(k-1)*6:SystemState*6+k*6,:)=u_test(:,i-1-ControlState+k);
    end
   % inputns(SystemState*6+1:SystemState*6+6,:)=u_test(:,i-1);
    InputNs=mapminmax('apply',inputns,inputps);
    
    for k = 1:size(segmentIndex,2)
              if(partitions.ifin(coeff(:,1:idx)'*(InputNs(1:SystemState*6,:)-mu'),segmentIndex{k},Dimension)==1)
                       Traj_switchN= ELMpredict(ELMs(k),InputNs);
                       Traj_switch(:,i)=mapminmax('reverse',Traj_switchN,outputps);
              end
    end
end
% 1.Plot Figures

%% Reachable set computation for our model 
% Ini_Input_u=[y_test(:,1);y_test(:,2)];
% Ini_Input_l=[y_test(:,1);y_test(:,2)];
% 
% Ini_SetInput=zeros(2*size(Ini_Input_l,1),1);
% 
% for i = 1:size(y_test,1)
%     Ini_Input_u(size(y_test,1)+i)=Ini_Input_l(size(y_test,1)+i)+delta;
% end
% 
% for i = 1:size(Ini_Input_l,1)
%     Ini_SetInput(2*(i-1)+1,1)= Ini_Input_l(i);
%     Ini_SetInput(2*i,1)=Ini_Input_u(i);
% end
% % Input torque: u_test(:,2)
% % Bound of Input torque
% ubound = zeros(size(u_test,1)*2,size(u_test,2));
% for i = 1:size(u_test,2)
%     for j = 1:size(u_test,1)
%         ubound(2*(j-1)+1,i)=u_test(j,i);
%         ubound(2*(j),i)=u_test(j,i);
%     end
% end
% %1. Single ELM model
% fprintf('Reachable set estimation for Single ELM using NNV.\n')
% 
% singleELMtime=zeros(durationReach,1);
% for i =1:1 % How many trajectories what to be examined
%     SELMbound{i}(:,1)= Ini_SetInput(1:2*size(y_test,1),:);
%     SELMbound{i}(:,2)= Ini_SetInput(2*size(y_test,1)+1:end,:);
%     for j = 3:durationReach
%         fprintf('step'); 
%        disp(j); 
%         tic
%         inputboundELM = ELMreachabilitynnv([SELMbound{i}(:,j-2);SELMbound{i}(:,j-1);ubound(:,j-1)],ELMs1);    
%         SELMbound{i}(:,j)=[inputboundELM];
%      toc
%      singleELMtime(j-1,1)=toc;
%     end
% end
% 
% switchELMtime=zeros(durationReach,1);
% %2. Switching ELM model  
% fprintf('Reachable set estimation for Single ELM using NNV.\n')
% for i=2:durationReach
% tic
% 
% switchELMtime(j-1,1)=toc;
% end



Traj_train=zeros(size(y_train,1),duration);
for i = 1:SystemState
    Traj_train(:,i)=[y_train(:,i)];
end
Traj_singlet=Traj_train;
Traj_switcht=Traj_train;
%inputN(:,1:3)= Traj

for i =SystemState+1:duration
    for j = 1:SystemState
        inputnt(1+(j-1)*6:j*6,:)= Traj_singlet(:,i-SystemState+j-1);
    end
    for k= 1:ControlState
    %   xs(i,j*6+1+(k-1)*6:j*6+k*6)=u_train(:,i+SystemState-k)';
       inputnt(SystemState*6+1+(k-1)*6:SystemState*6+k*6,:)=u_train(:,i-1-ControlState+k);
    end

    %inputnt(j*6+1:j*6+6,:)=u_train(:,i-1);

    InputN=mapminmax('apply',[inputnt],inputps);
    Traj_singleN=ELMpredict(ELMs1,InputN);
    Traj_singlet(:,i)=mapminmax('reverse',Traj_singleN,outputps); 
     
    for j = 1:SystemState
        inputnst(1+(j-1)*6:j*6,:)= Traj_switcht(:,i-SystemState+j-1);
    end
    for k= 1:ControlState
    %   xs(i,j*6+1+(k-1)*6:j*6+k*6)=u_train(:,i+SystemState-k)';
      inputnst(SystemState*6+1+(k-1)*6:SystemState*6+k*6,:)=u_train(:,i-1-ControlState+k);
    end

    %inputnst(6*j+1:6*j+6,:)=u_train(:,i-1);

    InputNs=mapminmax('apply',inputnst,inputps);
    
    for k = 1:size(segmentIndex,2)
              if(partitions.ifin(coeff(:,1:idx)'*(InputNs(1:SystemState*6,:)-mu'),segmentIndex{k},Dimension)==1)
                       Traj_switchN= ELMpredict(ELMs(k),InputNs);
                       Traj_switcht(:,i)=mapminmax('reverse',Traj_switchN,outputps);
              end
    end
end


for k_ax = 1:size(Traj_switch,1)
  Traj_switch(k_ax,:) = Traj_switch(k_ax,:) * y_std(k_ax) + y_mean(k_ax);
  Traj_switcht(k_ax,:)= Traj_switcht(k_ax,:) * y_std(k_ax) + y_mean(k_ax);
  Traj_single(k_ax,:) = Traj_single(k_ax,:) * y_std(k_ax) + y_mean(k_ax);
  Traj_singlet(k_ax,:) = Traj_singlet(k_ax,:) * y_std(k_ax) + y_mean(k_ax);
  y_train(k_ax,:) = y_train(k_ax,:) * y_std(k_ax) + y_mean(k_ax);
end



%%
figure
 for i = 1:size(y_test,1)
    subplot(size(y_test,1),1,i)
    plot(time_test(SystemState:duration),y_test(i,SystemState:duration))
    hold on 
    plot(time_test(SystemState:duration),Traj_single(i,SystemState:end))
    xlabel('time(s)')
    ylabel('position')
    hold on
 end
 title('single neural network modeling')
figure
for i = 1:size(y_test,1)
    subplot(size(y_test,1),1,i)
    plot(time_test(SystemState:duration),y_test(i,SystemState:duration))
    hold on 
    plot(time_test(SystemState:duration),Traj_switch(i,SystemState:end))
    xlabel('time(s)')
    ylabel('position')
    hold on
end
title('switching neural network modeling')

% 3.Linear model
% define ssest options
%  opt_ssest = ssestOptions ();
%  opt_ssest.Display = 'on';
%  opt_ssest.Focus = 'simulation';
% 
%  % zero initialize the input
%  utrain = u_train - u_train (:,1);
%  utest = u_test-u_test(:,1);
%  % define data
%  dt = 0.1;
%  id_data = iddata(y_train',utrain',dt);
%  % identify model
%  n_states = 12;
%  ss_model = ssest (id_data,n_states,opt_ssest );
%  compare(id_data,ss_model)
%  test_data=iddata(y_test',utest',dt);
%  figure
%  [y,fit,ic]=compare(test_data,ss_model)
%  compare(test_data,ss_model)

% 4. Compare from the stastical point of view

 %devia=std(y_test');

NRMSE_single=zeros(size(devia,1),1);
NRMSE_switch=zeros(size(devia,1),1);
ERR_single=zeros(size(devia,1),1);
ERR_switch=zeros(size(devia,1),1);
for i = 1:size(devia,1)
    for j = 1:size(Traj_single,2)
        ERR_single(i,1)=ERR_single(i,1)+(Traj_single(i,j)-y_test(i,j))^2;
        ERR_switch(i,1)=ERR_switch(i,1)+(Traj_switch(i,j)-y_test(i,j))^2;
    end
end
for i = 1:size(devia,1)
  NRMSE_single(i,1)=sqrt(1/size(Traj_single,2)*1/(devia(i,1))^2*ERR_single(i,1));
  NRMSE_switch(i,1)=sqrt(1/size(Traj_single,2)*1/(devia(i,1))^2*ERR_switch(i,1));
end


% 1.Plot Figures
figure
 for i = 1:size(y_train,1)
    subplot(size(y_train,1),1,i)
    plot(time_train(SystemState+1:duration),y_train(i,SystemState+1:duration))
    hold on 
    plot(time_train(SystemState+1:duration),Traj_singlet(i,SystemState+1:end))
    %xlabel('time(s)')
    %ylabel('position')
    hold on
 end
% title('single neural network modeling')
figure
for i = 1:size(y_train,1)
    subplot(size(y_train,1),1,i)
    plot(time_train(SystemState+1:duration),y_train(i,SystemState+1:duration))
    hold on 
    plot(time_train(SystemState+1:duration),Traj_switcht(i,SystemState+1:end))
    %xlabel('time(s)')
    %ylabel('position')
    hold on
end
%title('switching neural network modeling')
%%
t = tiledlayout(size(y_test,1),1,'TileSpacing','Compact');
t.TileSpacing = 'compact';
t.Padding = 'compact';
for i = 1:size(y_test,1)
    
nexttile
   % tsubplot(size(y_test,1),1,'tight')
    % set(gca,'position',[0.08,0.75,0.89,0.23])
    plot(time_train(SystemState+1:duration),y_train(i,SystemState+1:duration))
    hold on 
    plot(time_train(SystemState+1:duration),Traj_switcht(i,SystemState+1:end))
%    xlabel('time(s)')
%    ylabel('position')
%    hold on
 end



%% NRMSE of Simulation Mode
NRMSE_singlet=zeros(size(devia,1),1);
NRMSE_switcht=zeros(size(devia,1),1);
ERR_single=zeros(size(devia,1),1);
ERR_switch=zeros(size(devia,1),1);
for i = 1:size(devia,1)
    for j = 1:size(Traj_singlet,2)
        ERR_single(i,1)=ERR_single(i,1)+(Traj_singlet(i,j)-y_train(i,j))^2;
        ERR_switch(i,1)=ERR_switch(i,1)+(Traj_switcht(i,j)-y_train(i,j))^2;
    end
end
for i = 1:size(devia,1)
  NRMSE_singlet(i,1)=sqrt(1/size(Traj_singlet,2)*1/(devia(i,1))^2*ERR_single(i,1));
  NRMSE_switcht(i,1)=sqrt(1/size(Traj_singlet,2)*1/(devia(i,1))^2*ERR_switch(i,1));
end

disp('NRMSE_switch')
mean(NRMSE_switcht)

%% NRMSE of Prediction Mode
NRMSE_singlePre=zeros(size(devia,1),1);
NRMSE_switchPre=zeros(size(devia,1),1);
ERR_single=zeros(size(devia,1),1);
ERR_switch=zeros(size(devia,1),1);
for i = 1:size(devia,1)
    for j = 1:size(ON,2)
        ERR_single(i,1)=ERR_single(i,1)+(ON(i,j)-y_test(i,j+SystemState))^2;
        ERR_switch(i,1)=ERR_switch(i,1)+(ON_S(i,j)-y_test(i,j+SystemState))^2;
    end
end
for i = 1:size(devia,1)
  NRMSE_singlePre(i,1)=sqrt(1/size(ON,2)*1/(devia(i,1))^2*ERR_single(i,1));
  NRMSE_switchPre(i,1)=sqrt(1/size(ON,2)*1/(devia(i,1))^2*ERR_switch(i,1));
end

disp('NRMSE_switch')
mean(NRMSE_switcht)