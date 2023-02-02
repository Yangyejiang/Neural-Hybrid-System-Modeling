clear
close all
clc
names = {'Angle','CShape','LShape','PShape','Spoon'};
figName = [];

fprintf('\nAvailable Models:\n')
for i=1:size(names,2)
         fprintf(names{i})
          fprintf('\n')
end
n = input('\nType the number of the model you wish to run [type 0 to exit]: ');
    fprintf('\n\n')
   if (n<0 || n>size(names,2))
        disp('Wrong model number!')
        disp('Please try again and type a number between 1-30.')
    elseif n == 0
        return
    else
      load(['DataSet/' names{n}],'demos','dt') %loading the model
   end
      
% How many trajectories we want to obtain
NeuronNum_switch=40;
NeuronNum_single=200;
% Sampling data from working zone
u=0.2;
delta=0.5;
tau=0.01;

duration = 1000;
e=1.8e-4;
tol = 5;
maximum_entropy=40;
Dimension=2;
% Initial Input Set
durationReach=20;

switch(n)
    case 1
        inputbound =[-44.35;-44.3;-3.12;-3.1];
        thetaInterval=[-46 -44;-3.1 -2];
    case 2
         thetaInterval=[-1.3 2;38 40];
         inputbound = [-0.2;0;39.8;40];
    case 3
          thetaInterval=[-31.5 -31;41.5 43];
          inputbound =[-31.2;-31;42.8;42.9];
    case 4
        inputbound =[-18.02;-18;-18.02;-18];
        thetaInterval=[-19.5 -19;-19.5 -19];
    case 5
        inputbound =[-47.02;-47;0;0.2];
        thetaInterval=[-48 -44;0 0.6];
end
%% Load Data Set

for i = 1:size(demos,2)
    TrajData{i} = demos{i}.pos;
    plot(TrajData{i}(1,:),TrajData{i}(2,:))
    hold on
end
%
TrajNum=size(demos,2);

%Extract Data
xs=zeros((size(TrajData{1},1))*TrajNum,2);
t=zeros((size(TrajData{1},1))*TrajNum,2);
for i = 1:TrajNum
    Begin=(size(TrajData{1},2)-1)*(i-1)+1;
    End = (size(TrajData{1},2)-1)*(i-1)+size(TrajData{1},2);
    xs(Begin:End-1,1) = TrajData{i}(1,1:end-1);
    xs(Begin:End-1,2) = TrajData{i}(2,1:end-1);
    t(Begin:End-1,1) = TrajData{i}(1,2:end);
    t(Begin:End-1,2) = TrajData{i}(2,2:end);
end

% Find the bounderies
lowerbound= min(t)-0.5 ;
upperbound= max(t)+0.5;
init_interval{1}=[lowerbound',upperbound'];

%% Data-driven Partitioning
P=partitions(init_interval,xs,t);
intervals=ME(P,tol,maximum_entropy,Dimension);
P.intervals=intervals;


%% Initialized ELM and Merge Partitions
ELMs1=ELM.GenerateELM(size(xs,2),NeuronNum_switch,'ReLu',size(t,2));
[P1,ELMs]=MergePatitions(P,ELMs1,e);

% Plot intervals
figure
partitions.intervalplot(P.intervals,'empty','black')
figure
partitions.intervalplot(P.intervals,'empty','black')
partitions.intervalplot(P1.intervals,'full','red')
%title('Invariantspace using Bisection method')

%% Train a Complex Neural Network Model as referance
ELMs1=ELM.GenerateELM(size(xs,2),NeuronNum_single,'ReLu',2);
ELMs1=trainELM(ELMs1,xs',t');

%% Verfify whether can it approximate the dynamics well
TestTraj = 50;%number of test trajectories
%Randomly generate staring state
%thetaInterval=[floor(min(xs(:,1))) ceil(max(xs(:,1)));floor(min(xs(:,2))) ceil(max(xs(:,2)))];
RandomStateInput = intervalCompute.randomPoint(thetaInterval,TestTraj);
%Random start state should be within the select input space
j=1;
segmentIndex=P1.intervals;
inputspace1=P1.intervals;
for i=1:TestTraj
    for k = 1:size(segmentIndex,2)
        for z = 1:size(segmentIndex{k},1)/2
              if(RandomStateInput(1,i)>segmentIndex{k}(2*z-1,1))&&(RandomStateInput(1,i)<segmentIndex{k}(2*z-1,2))
                    if(RandomStateInput(2,i)>segmentIndex{k}(2*z,1))&&(RandomStateInput(2,i)<segmentIndex{k}(2*z,2))
                       Traj{1,j}=RandomStateInput(:,i);
                    end
              end
        end
    end
    j=j+1;
end
validNum=size(Traj,2);
%Generate the system response and break while it's out of input space

% 1.Input uk from uniform distribution

for j=1:size(Traj,2)
  %  uk = zeros(size(2:1:duration,2)+1,1);
  % uk = u*ones(size(uk,1),1)-1+(1-(-1))*rand(size(uk,1),1);
    for i = 2:1:duration
         flag=0;
       for k=1:size(segmentIndex,2)
           for z= 1:size(segmentIndex{k},1)/2
              if(Traj{1,j}(1,i-1)>segmentIndex{k}(2*z-1,1)&&((Traj{1,j}(1,i-1)<segmentIndex{k}(2*z-1,2)))&&(flag==0))
                    if(Traj{j}(2,i-1)>segmentIndex{1,k}(2*z,1))&&((Traj{j}(2,i-1)<segmentIndex{1,k}(2*z,2)))%state dependent switch
  %                      Traj{1,j}(3,i-1) = uk(i-1);                    
                        Traj{1,j}(1:2,i)= ELMpredict(ELMs(k),Traj{1,j}(:,i-1));
                        TrajflagDDM{j}(i-1)=k; 
                        flag=1;
                     end
              end
             if(k==size(inputspace1,2))&&(flag==0) 
                  TrajEndflag(j)=1; 
                  j=j+1;
                  i=2;
              break;
             end
           end
        end
    end  
end


% For Reference model

% for i =1:validNum
%   TrajRe{i}=[Traj{1,i}(1,1) ;Traj{1,i}(2,1)];
%       for j = 2:1:duration
%          flag=0;
% %         uin=uk(j-1,1);
%          TrajRe{i}(1,j)=(1+tau)*TrajRe{i}(1,j-1)-tau*TrajRe{i}(1,j-1)^3+tau*uin;
%          TrajRe{i}(2,j)=TrajRe{i}(2,j-1)+2*pi/3*tau;
%       end
% end
% for i =1:validNum
%  TrajRe{i} = [TrajRe{1,i}(1,:).*cos(TrajRe{1,i}(2,:)) ;TrajRe{1,i}(1,:).*sin(TrajRe{1,i}(2,:))];
% end

TrajELMs=Traj;

% For Single model

for j=1:validNum
    for i = 2:1:duration              
             % TrajELMs{1,j}(3,i-1) = uk(i-1,1);
              TrajELMs{1,j}(1:2,i)= ELMpredict(ELMs1,TrajELMs{1,j}(:,i-1));
%               TrajBPs{1,j}(1:2,i)= BPs([TrajBPs{1,j}(:,i-1);uk(i-1,j)]);
    end
end

% Plot figures
figure
for i=1:validNum
    plot(Traj{1,i}(1,:),Traj{1,i}(2,:))
    hold on
end
title('DataDriven State Denpendent Swtiching Model')
xlabel('\theta_1')
ylabel('\theta_2')
grid on

% figure
% for i=1:validNum
%     plot(TrajRe{1,i}(1,:),TrajRe{1,i}(2,:))
%     hold on
% end
% title('Simple Limit Cycle Model')
% xlabel('\theta_1')
% ylabel('\theta_2')
% grid on

figure
for i=1:validNum
    plot(TrajELMs{1,i}(1,:),TrajELMs{1,i}(2,:))
    hold on
end
title('Single ELM Model')
xlabel('\theta_1')
ylabel('\theta_2')

%% Analysis of Set-valued Reachability

%1.Single Neural Network as Data Driven Model
fprintf('Reachable set estimation for Single ELM using NNV.\n')
singleELMtime=zeros(199,1);
for i =1:1 % How many trajectories what to be examined
    SELMbound{i}(:,1)=[inputbound];
%    SBPbound{i}(:,1)=[inputbound;ubound];
%     Reachablity Analysis for Data Driven Model 
    for j = 2:durationReach
        fprintf('step'); 
       disp(j); 
        tic
        inputboundELM = ELMreachabilitynnv(SELMbound{i}(:,j-1),ELMs1);    
 %       inputboundBP  = BPreachabilitynnv(SBPbound{i}(:,j-1),BPs);
        SELMbound{i}(:,j)=[inputboundELM];
 %       SBPbound{i}(:,j)=[inputboundBP;ubound];
     toc
     singleELMtime(j-1,1)=toc;
    end

end

% 2. Switching Neural networks as Data Driven Model
switchELMtime=zeros(199,1);
fprintf('Reachable set estimation for switching ELM time using nnv')
    for i = 1:1
         ELMbound{i}(:,1)=inputbound;     
        %Reachability Anlysis for Multiple Neural Network
        for j = 2:durationReach
     fprintf('step'); 
       disp(j);           
                   
            counter=1;
       % a.split
[splittingset,time]=partitions.SplitingReach(P1.intervals,ELMs,ELMbound{i}(:,j-1),'nnv');
                % 2.combine
       fprintf('takes %d',time)
       fprintf(' second \n')
                switchELMtime(j-1,1)=time;
                ELMbound{i}(1:4,j) = partitions.CombineReach(splittingset);
        end
    end

%% Plot Interval sets
%1.Convert the input vector to output cells
%     for single ELM
    for i = 1:size(SELMbound{1,1},2)-1
        singleELM.bound{1,i}(1,1:2)=SELMbound{1,1}(1:2,i)';
        singleELM.bound{1,i}(2,1:2)=SELMbound{1,1}(3:4,i)';
    end
   
%    title({['Input intervals based on simulation-guided method of Single ELM'];[num2str(size(SELMbound{1,1},2)), ' Input intervals']})
%     xlabel('x_1') 
%     ylabel('x_2')
%     axis([-4 2 -4 2]); 
%     set(gca, 'GridLineStyle', ':');
%     set(gca, 'GridAlpha', 1);
%     set(gca, 'XTick', -4:2/1:4);
%     set(gca, 'YTick', -4:2/1:4);
%     grid on
    %For Switching ELM
    k=1;
    for i = 1:size(ELMbound{1,1},2)
        switchingELM.bound{1,i}(1,1:2)=ELMbound{1,1}(1:2,i)';
        switchingELM.bound{1,i}(2,1:2)=ELMbound{1,1}(3:4,i)';
    end
 %   figure

         figure
    partitions.intervalplot(switchingELM.bound,'empty','b')
    hold on
 
    partitions.intervalplot(singleELM.bound,'empty','m')
    %    title({['Input intervals based on simulation-guided method of Switching ELM'];[num2str(size(ELMbound{1,1},2)), ' Input intervals']})
    xlabel('x_1') 
    ylabel('x_2') 
    % axis([-1.5 2 -1.5 2]); 
     grid on
     set(gca, 'GridLineStyle', ':');
     set(gca, 'GridAlpha', 1);
    set(gca, 'XTick', -4:1/1:4);
     set(gca, 'YTick', -4:1/1:4);
     set(gca, 'XMinorGrid','on');
     set(gca, 'YMinorGrid','on');