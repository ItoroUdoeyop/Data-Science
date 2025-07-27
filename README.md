
# 💎 Gemstone Price Prediction with Deep Learning

This project demonstrates how to use a **deep neural network** to predict gemstone prices based on numerical features. Built with **TensorFlow and Keras**, the model learns from a synthetic dataset to estimate gem values accurately using regression techniques.

## 🎯 Project Objective

To build and train a deep learning model that can **predict the price of a gemstone** using features such as:

- `feature1`
- `feature2`

These represent gem-specific numerical characteristics (e.g., carat, weight, dimensions).

## 🧠 Model Overview

- **Architecture**:
  - Fully connected neural network (Sequential API)
  - Three hidden layers with 4 neurons each, using ReLU activation
  - Output layer with a single neuron (for regression)
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: RMSprop
- **Training**: 250 epochs

## 📊 Workflow

1. **Data Preparation**:
   - Load data 
   - Normalize using `MinMaxScaler`
   - Split into training and testing sets

2. **Model Training**:
   - Deep Neural Network trained on the scaled data
   - Trained for 250 epochs to minimize MSE

3. **Evaluation**:
   - Visualize training loss
   - Predict and compare on test set
   - Evaluate using MAE, MSE, and RMSE

4. **Deployment**:
   - Model saved using Keras `.save()` method
   - Loaded and reused for predictions on new data

## 🔧 Tech Stack

- Python 3.x
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn
- TensorFlow / Keras

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/gemstone-price-predictor-dl.git
   cd gemstone-price-predictor-dl
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook gem_model.ipynb
   ```

## 📁 File Structure

```
gemstone-price-predictor-dl/
├── gem_model.ipynb       # Main notebook with data loading, training, and evaluation
├── TensorFlow_FILES/     # Contains the CSV dataset
├── my_gem_model.keras    # Saved trained model
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
```

## 📌 Example Use Case

After training, the model can predict the price of a new gemstone:
```python
new_gem = [[998, 1000]]
new_gem = scaler.transform(new_gem)
model.predict(new_gem)
```

## 🤝 Contributions

Open to pull requests and issues! Suggestions to expand the dataset or add model tuning are welcome.
#  Gemstone_Price_Prediction_with_Deep_Learning
