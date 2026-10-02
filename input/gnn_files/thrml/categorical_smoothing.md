# GNN Example: THRML categorical smoothing

## GNNSection
ActInfPOMDP

## GNNVersionAndFlags
GNN v1

## ModelName
THRML Positive Categorical Smoothing

## ModelAnnotation
Three hidden states, two observations and two fixed transition actions.
Every conditional has positive support. THRML estimates full-sequence
smoothing by categorical Gibbs sampling. C and E remain source preferences
and habits; this experimental inference adapter does not optimize actions.

## StateSpaceBlock
s[3,1,type=float]
o[2,1,type=int]
A[2,3,type=float]
B[3,3,2,type=float]
C[2,type=float]
D[3,type=float]
E[2,type=float]
u[1,type=int]
t[1,type=int]

## Connections
D>s
s-A
A-o
s-B
B>u

## InitialParameterization
A={(0.75,0.15,0.35),(0.25,0.85,0.65)}
B={((0.7,0.2),(0.1,0.6),(0.3,0.1)),((0.2,0.3),(0.6,0.1),(0.2,0.2)),((0.1,0.5),(0.3,0.3),(0.5,0.7))}
C={(0.7,-0.2)}
D={(0.2,0.3,0.5)}
E={(0.3,0.7)}

## Time
Time=t
Dynamic
Discrete
ModelTimeHorizon=3

## ModelParameters
num_hidden_states: 3
num_obs: 2
num_actions: 2
num_timesteps: 3
b_tensor_order: next_state_previous_state_action

## ActInfOntologyAnnotation
A=LikelihoodMatrix
B=TransitionMatrix
C=LogPreferenceVector
D=PriorOverHiddenStates
E=HabitPrior
s=HiddenState
o=Observation
u=Action

## Footer
Use explicit THRML options observations=[0,1,0], transition_actions=[1,0]
for the independently checked smoothing witness. Without observations the
adapter records a synthetic THRML Gibbs joint draw; fixed action defaults to 0.
