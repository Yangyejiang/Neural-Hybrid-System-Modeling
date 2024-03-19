# Neural-Hybrid-System-Modeling (NSHM)
Matlab Toolbox for Distributed Neural Hybrid System Modeling 

This toolbox implements data-driven processes including PCA featuring; ME bisecting; and Merging, in modeling complex dynamical systems, particularly focusing on increasing the learning models' scalability.

# Related Tools and Software
This toolbox makes use of the Neural Network Verification (NNV) for reachability analysis; GNU Linear Programming Kit (glpkmex) and YALMIP in training the neural networks.

# Installation

1) Install MATLAB

2) Clone or download the NNV toolbox from (https://github.com/Yangyejiang/Neural-Hybrid-System-Modeling)

3) Open Matlab, then add a directory to the root directory where NSHM exists on your machine

# Running tests and examples

Go into the Neural-Hybrid-System-Modeling/examples` folders to execute the scripts for testing/analyzing examples.

# Features
This tool uses a data-driven method to enhance the scalability of neural network modeling.

1. Bisecting the feature space using ME partitioning\
![image](https://github.com/aicpslab/Neural-Hybrid-System-Modeling/blob/main/Example/ME%20bisecting.png)
2. Merging and Dynamics Learning\
![image](https://github.com/aicpslab/Neural-Hybrid-System-Modeling/blob/main/Example/Merging%20and%20Learning%20of%20NHS.png)
3. Reachable Set Computation and Counting\
![image](https://github.com/aicpslab/Neural-Hybrid-System-Modeling/blob/main/Example/ReachableSetComputation.png)

# Complex Dynamical System Modeling

The case study of a six-axis industrial robot arm, drawing upon the dataset and baseline established in the research conducted by J. Weigand, J. Götz, J. Ulmen, and M. Ruskowski, titled "Dataset and Baseline for an Industrial Robot Identification Benchmark", accessible at https://www.nonlinearbenchmark.org/benchmarks/industrial-robot, is given as an example of the complex dynamical system modeling. \
1. Conventional Single Neural Network Model\
![image](https://github.com/aicpslab/Neural-Hybrid-System-Modeling/blob/main/Example/Fig/Single_IndustrialSimulation.png)

2. Our Proposed Distributed Learning Model\
![image](https://github.com/aicpslab/Neural-Hybrid-System-Modeling/blob/main/Example/Fig/Switch_IndustrialSimulation.png)
Our distributed learning example (Neural-Hybrid-System-Modeling/examples/Industrial_Robot.m) achieved high accuracy in reconstructing the robot arm's dynamics. Notably, our model demonstrated superior performance during the low-velocity phase compared to not only the conventional neural network modeling method but also the original modeling approach (Neural-Hybrid-System-Modeling/examples/Industrial_ReferenceModel_Forward.m) (this reference example is based on https://kluedo.ub.rptu.de/frontdoor/deliver/index/docId/7284/file/Robot_Identification_Benchmark_Description.pdf).\
3. Reference Model\ 
![image](https://github.com/aicpslab/Neural-Hybrid-System-Modeling/blob/main/Example/Fig/ReferenceModel.png)
