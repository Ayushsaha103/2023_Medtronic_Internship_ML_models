# 2023 Medtronic Internship - My Machine Learning Models
Machine Learning models I implemented (and built from scratch) to model the activity of their next-generation products in various use conditions
The accuracy of these tools ranged from 90-98%, and have informed our team about the products' ability to meet product compliance documentation standards.

# Gaussian Process Regressor (GPR)
- Purpose: This model learns to predict an independent variable based off an input vector of data features, and additionally predicts its own uncertainty in its guess.
- Advantages: This allows for it to easily draw predictive probability distributions for given input vectors, and it can model a limited amount of data with decent accuracy.
- Disadvantages: Inoptimal for large datasets, and requires 3 hyper-parameters to be manually determined/initialized

- Credits: to Yoon et. al. for the mathematical formulas: https://www.researchgate.net/publication/351269804_Interaction-Aware_Probabilistic_Trajectory_Prediction_of_Cut-In_Vehicles_Using_Gaussian_Process_for_Proactive_Control_of_Autonomous_Vehicles

### My GPR model's prediction results:
![image](https://github.com/Ayushsaha103/2023_Medtronic_Internship_ML_models/assets/71895904/2d5badc9-ac70-4693-ab01-498feee87f99)

# Random Forest Regressor
![image](https://github.com/Ayushsaha103/2023_Medtronic_Internship_ML_models/assets/71895904/2165f8d8-af7c-4329-b4fd-d25dcf9c5f00)

- Purpose: This model learns to generate decision trees which model how input data feature vectors are related to an independent (y) variable.
- How it works: When fed an input feature vector, each decision tree reaches a single leaf nodel. The algorithm summarizes the output node value from all the trees when determining its prediction for the independent variable.
- Learn more here: (credits to Josh Stammer): https://www.youtube.com/watch?v=g9c66TUylZ4&pp=ygUlam9zaCBzdGFtbWVyIHJhbmRvbSBmb3Jlc3QgcmVncmVzc2lvbg%3D%3D

### My Random Forest model's prediction results:
![image](https://github.com/Ayushsaha103/2023_Medtronic_Internship_ML_models/assets/71895904/3e4914bb-fd6e-4ddf-8db6-c336ddcb1d9d)

### Bayesian Neural Network
- Purpose: (Same as GPR): This model learns to predict an independent variable based off an input vector of data features, and additionally predicts its own uncertainty in its guess.
- Advantages: (Same as GPR), add-on: This model works efficiently and quickly for larger datasets
- Disadvantage: Slower to train than a normal neural network

- Credits: to Oduerr for the tutorial: https://github.com/tensorchiefs/dl_book/blob/master/chapter_05/nb_ch05_01.ipynb




