% Setup program AOCS Reinforcement learning framwork
% Create AOCS Environment Interface with Bus

% Specify the model

mdl = 'rlAOCSModel_one_agent_2022_03_09_reduced_q';
% open_system(mdl);

% Specify the path to the agent block.
blks = mdl + "/TD3 Agent for AOCS";

% Constants
qe_max = 0.18;
qe_min = 0.09;
omega_max = 0.015;

% Create the action and observation specification objects.

numobs = 13;
numAct = 3;
lm = -ones(numobs,1);
obsInfo = rlNumericSpec([numobs 1],...
    'LowerLimit',lm,...
    'UpperLimit',-lm);
obsInfo.Name = 'observations';
obsInfo.Description = 'Q, Q_error, W, W_error';

actInfo = rlNumericSpec([numAct 1],...
    'LowerLimit',-0.012*ones(numAct,1),...
    'UpperLimit',0.012*ones(numAct,1));
actInfo.Name = 'tau';
actInfo.Description = 'tau_X, tau_Y, tau_Z';

% Create the reinforcement learning environment for spacecraft AOCS.

env = rlSimulinkEnv(mdl,blks,obsInfo,actInfo);

% Reset function 
env.ResetFcn = @(in)localResetFcn(in);

% Specify the simulation time Tf and the agent sample time Ts in seconds.

Ts = 0.25;
Tf = 60;

% Fix the random generator seed for reproducibility.

rng(0)
maxsteps = ceil(Tf/Ts);

% Create TD3 Agent

% Agent options and creation
Ts_agent = Ts;
agentOptions = rlTD3AgentOptions("SampleTime",Ts_agent, ...
    "DiscountFactor", 0.995, ...
    "SaveExperienceBufferWithAgent",false,...
    "ResetExperienceBufferBeforeTraining",false,...
    "ExperienceBufferLength",1e6, ...
    "MiniBatchSize",maxsteps, ...
    "NumStepsToLookAhead",1, ...
    "TargetSmoothFactor",0.01, ...
    "TargetUpdateFrequency",1);

kernal_value = ones(3,1);
agentOptions.ExplorationModel = rl.option.OrnsteinUhlenbeckActionNoise;
agentOptions.ExplorationModel.StandardDeviation = 30e-5*kernal_value;
% agentOptions.ExplorationModel.StandardDeviation = 24e-5*kernal_value;
% agentOptions.ExplorationModel.StandardDeviation = 1e-3*kernal_value;
agentOptions.ExplorationModel.StandardDeviationDecayRate = 0*kernal_value;
% agentOptions.ExplorationModel.StandardDeviationDecayRate = 5e-6*kernal_value;
% agentOptions.ExplorationModel.StandardDeviationDecayRate = 2e-6*kernal_value;
agentOptions.ExplorationModel.StandardDeviationMin = 0*kernal_value;
agentOptions.TargetPolicySmoothModel.Variance = 0.1;
agentOptions.TargetPolicySmoothModel.VarianceDecayRate = 1e-4;
% 
TD3_AOCS = rlTD3Agent(getActor(obsInfo,numobs,actInfo,numAct),...
                    getCritic(obsInfo,numobs,actInfo,numAct),...
                    agentOptions);

% Agent training options

maxepisodes = 501;

% trainOpts = rlTrainingOptions(...
%     'MaxEpisodes',maxepisodes,...
%     'MaxStepsPerEpisode',maxsteps,...
%     'ScoreAveragingWindowLength',10,...
%     'Verbose',false,...
%     'SaveAgentCriteria','EpisodeCount',...
%     'SaveAgentValue',1,...
%     'SaveAgentDirectory','C:\Users\Public\Documents\MATLAB',...
%     'Plots','training-progress',...
%     'StopTrainingCriteria','AverageReward',...
%     'StopTrainingValue',1e6);

trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes,...
    'MaxStepsPerEpisode',maxsteps,...
    'ScoreAveragingWindowLength',50,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',1e6);

% Agent training%

w=warning('off','all');
% tr_noshaping_rewQ_obsq = train(TD3_AOCS,env,trainOpts);
tr_gauss_5em5_500ep_g20_rewQ_obsq = train(TD3_AOCS,env,trainOpts);
% tr_none_500ep_g20_rewQ_obsq = train(TD3_AOCS,env,trainOpts);

% Reset function
function in = localResetFcn(in)
Wmax = evalin('base','omega_max');
mdl = evalin('base','mdl');
% tr = evalin('base','trainingStats');
persistent numep
  if isempty(numep)
        numep = 0;
  end
  
%1- Randomely generate Q_init for REF generation and SAT kinematics
Q = 2*rand(1,4)-(ones(1,4));
Q = quatnormalize(Q);
blk = mdl +"/Q";
in = setBlockParameter(in,blk,'Value',mat2str(Q));
Q0 = Q + (0.01*(rand(1,4)-(.5*ones(1,4))));
Q0 = quatnormalize(Q0);
blk = mdl +"/Q0";
in = setBlockParameter(in,blk,'Value',mat2str(Q0));

%2- Randomely generate W_ref (REF generation) and W0 for SAT dynamics
W = (rand([3 1])-(.5*ones(3,1)))*2*Wmax;
blk = mdl +"/W";
in = setBlockParameter(in,blk,'Value',mat2str(W));

W0 = W + .005*(rand([3 1])-(.5*ones(3,1)));
W0 = max(-Wmax*ones(3,1),W0);
W0 = min(Wmax*ones(3,1),W0);
blk = mdl +"/W0";
in = setBlockParameter(in,blk,'Value',mat2str(W0));    
blk = mdl +"/CLK";
% disp (numep)
in = setBlockParameter(in,blk,'Value',mat2str(numep));
numep = numep + 1;
end
function actor = getActor(observationInfo,numObservations,actionInfo,numActions)%,hiddenLayerSize)
init = 'zeros';%'glorot';%;%'narrow-normal';%'zeros';
actorNetwork = [
    featureInputLayer(numObservations,'Normalization','none','Name','State')
    fullyConnectedLayer(64, 'Name','actorFC1')%,'WeightsInitializer',init)
    tanhLayer('Name','relu1')
    fullyConnectedLayer(32, 'Name','actorFC2')%,'WeightsInitializer',init)
    tanhLayer('Name','relu2')
    fullyConnectedLayer(16, 'Name','fc4')%,'WeightsInitializer',init)
    tanhLayer('Name','relu3')
    fullyConnectedLayer(numActions,'Name','Action')%,'WeightsInitializer',init)
    tanhLayer('Name','tau')];
actorOptions = rlRepresentationOptions('LearnRate',3e-3,'GradientThreshold',5,'L2RegularizationFactor',0.01);
actor = rlDeterministicActorRepresentation(actorNetwork,observationInfo,actionInfo,...
    'Observation',{'State'},'Action',{'tau'},actorOptions);
end

function critic = getCritic(observationInfo,numObservations,actionInfo,numActions)%,hiddenLayerSize)
init = 'glorot';%'narrow-normal';%'zeros';
statePath = [
    featureInputLayer(numObservations,'Normalization','none','Name','State')
    fullyConnectedLayer(32,'Name','fc1')];%,'WeightsInitializer',init)];
actionPath = [
    featureInputLayer(numActions, 'Normalization', 'none','Name','Action')
    fullyConnectedLayer(32, 'Name','fc2')];%,'WeightsInitializer',init)];
commonPath = [additionLayer(2,'Name','add')
    reluLayer('Name','relu2')
    fullyConnectedLayer(16, 'Name','fc4')%,'WeightsInitializer',init)
    reluLayer('Name','relu4')
    fullyConnectedLayer(1, 'Name','CriticOutput')%,'WeightsInitializer','zeros')
    reluLayer('Name','relu5')];
criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,'fc1','add/in1');
criticNetwork = connectLayers(criticNetwork,'fc2','add/in2');

criticOptions = rlRepresentationOptions('LearnRate',1e-2,'GradientThreshold',5);
critic1 = rlQValueRepresentation(criticNetwork,observationInfo,actionInfo,...
    'Observation',{'State'},'Action',{'Action'},criticOptions);
critic2 = rlQValueRepresentation(criticNetwork,observationInfo,actionInfo,...
    'Observation',{'State'},'Action',{'Action'},criticOptions);

critic = [critic1,critic2];

end