# 2023 Medtronic Internship - My Machine Learning Models
Machine Learning models I implemented (and built from scratch) to probabalistically model the behavior of an upcoming Medtronic product in various use conditions
The accuracy of these tools ranged from 90-98%. These models have informed our team about the products' ability to meet product compliance standards.

# Gaussian Process Regressor (GPR)
- Purpose: This model learns to predict an independent variable based off an input vector of data features, and additionally predicts its own uncertainty in its guess.
- Advantages: This allows for it to easily draw a probability distribution which informs us of an interval of values where the independent variable may lie, corresponding to each given input feature vector. The GPR can model a limited amount of data with decent accuracy.
- Disadvantages: The GPR is inoptimal for large datasets, and requires 3 hyper-parameters to be manually determined/initialized

- Credits: to Yoon et. al. for the mathematical formulas: https://www.researchgate.net/publication/351269804_Interaction-Aware_Probabilistic_Trajectory_Prediction_of_Cut-In_Vehicles_Using_Gaussian_Process_for_Proactive_Control_of_Autonomous_Vehicles

### My GPR model's prediction results:
![image](https://github.com/Ayushsaha103/2023_Medtronic_Internship_ML_models/assets/71895904/2d5badc9-ac70-4693-ab01-498feee87f99)

# Bayesian Neural Network
![image](https://github.com/Ayushsaha103/2023_Medtronic_Internship_ML_models/assets/71895904/90c2174b-5212-43f3-8a6f-581c8e73957c)

- Purpose: (Same as GPR): This model learns to predict the value of the independent variable, based off an input vector of data features, and additionally predicts its own uncertainty in its guess.
- Advantages: (Same as GPR), add-on: This model works more efficiently than a GPR for larger datasets
- Disadvantage: Slower to train than a normal neural network

- Credits: to Oduerr for the tutorial: https://github.com/tensorchiefs/dl_book/blob/master/chapter_05/nb_ch05_01.ipynb

# Random Forest Regressor
![image](https://github.com/Ayushsaha103/2023_Medtronic_Internship_ML_models/assets/71895904/2165f8d8-af7c-4329-b4fd-d25dcf9c5f00)

- Purpose: This model learns to generate decision trees which model how input data feature vectors are related to an independent variable.
- How it predicts a test variable: When fed an input feature vector, each of the model's decision trees is traversed, and each ends up at a single leaf node (which represents its decided value for the independent variable). The algorithm takes the average of all of these leaf nodes' values from the trees in order to establish a single prediction for the value of the independent variable.
- Learn more here: (credits to Josh Stammer): https://www.youtube.com/watch?v=g9c66TUylZ4&pp=ygUlam9zaCBzdGFtbWVyIHJhbmRvbSBmb3Jlc3QgcmVncmVzc2lvbg%3D%3D

### My Random Forest model's prediction results:
![image](https://github.com/Ayushsaha103/2023_Medtronic_Internship_ML_models/assets/71895904/3e4914bb-fd6e-4ddf-8db6-c336ddcb1d9d)





