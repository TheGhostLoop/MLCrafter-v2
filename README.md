# 🤖 MLCrafter v2

**MLCrafter v2** is an intelligent terminal-based AutoML tool that simplifies machine learning workflows. Train, evaluate, and deploy custom **Classification** and **Regression** models from CSV datasets with zero coding required. Built with `scikit-learn` and designed for both beginners and data scientists.

---

## ✨ Key Features

### 🧠 **Smart Model Training**
* 📁 **Universal CSV Support** - Load any structured dataset
* 🎯 **Auto Problem Detection** - Intelligently identifies Classification vs Regression
* 🔧 **Automated Preprocessing** - Handles missing values, duplicates, and encoding
* 📊 **Dynamic Feature Selection** - Uses `SelectKBest` with mutual information
* 🏆 **Multi-Model Competition** - Trains multiple algorithms and picks the best

### 🔮 **Flexible Predictions**
* ⌨️ **Terminal Input** - Enter data manually for quick predictions  
* 📂 **Batch CSV Predictions** - Upload CSV files for bulk predictions
* 🧩 **Smart Data Handling** - Automatically manages categorical encoding
* 📈 **Probability Scores** - Shows confidence for classification tasks

### 📊 **Advanced Visualization**
* 📉 **Interactive Plots** - Actual vs Predicted comparisons
* 🎛️ **User-Chosen Features** - Select which feature to visualize
* 📊 **Performance Metrics** - R², Accuracy, MAE, RMSE display
* 🎨 **Professional Styling** - Publication-ready graphs

### 💾 **Model Management**
* 🗃️ **Complete Model Bundles** - Saves pipelines, scalers, and metadata
* 🔄 **Load & Resume** - Reuse trained models instantly  
* 📋 **Model Library** - Organized storage in `savedmodels/`

---

## 📂 Project Structure
```
MLCrafter-v2/
├── 📁 savedmodels/          # Trained model bundles (.joblib)
├── 📁 csvfiles/             # Sample datasets for testing
├── 🐍 main.py               # Main application entry point
├── 📋 requirements.txt      # Python dependencies
└── 📖 README.md             # Documentation (you're here!)
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.7+ 
- pip package manager

### Quick Install
```bash
# Clone or download the project
cd MLCrafter-v2

# Install dependencies
pip install -r requirements.txt
```

### Dependencies (`requirements.txt`)
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
joblib>=1.1.0
tkinter  # Usually included with Python
```

---

## 🚀 Getting Started

### Launch the Application
```bash
python main.py
```

### Main Menu Options
```
🄼🄻 🄲🅁🄰🄵🅃🄴🅁 v2
----------------------
1 -> Train a new model
2 -> Use trained model  
3 -> Exit
```

---

## 📖 Complete Workflow Guide

### 🔧 **Training a New Model**

1. **Data Loading**
   ```
   Selected DataSet: housing_data.csv
   Valid Rows: 1000
   Valid Columns: 6
   ```

2. **Target Selection**
   ```
   1 -> Size(SqFt)
   2 -> Bedrooms  
   3 -> Bathrooms
   4 -> Age(Years)
   5 -> DistanceFromCity(KM)
   6 -> Price
   Enter Column No. -> 6
   ```

3. **Auto Problem Detection**
   ```
   Now Model Will Predict -> Price
   [✔] Detected Problem Type: Regression
   ```

4. **Model Training & Evaluation**
   ```
   Dataset Trained On: RandomForestRegressor
   R² Score: 0.9847
   Mean Absolute Error: 15420.32
   Root Mean Squared Error: 23156.78
   ```

### 🔮 **Making Predictions**

#### Option 1: Manual Input
```
Choose prediction method:
1 -> Manual Input (Terminal)
2 -> CSV File Input
Enter choice: 1

No. Many Predictions: 2
Enter Data For 1 Row:
Size(SqFt): 1200
Bedrooms: 3
Bathrooms: 2
Age(Years): 5
DistanceFromCity(KM): 8

Price #1 Result: 625000.0
Price #2 Result: 580000.0
```

#### Option 2: CSV Batch Processing
```
Choose prediction method:
1 -> Manual Input (Terminal) 
2 -> CSV File Input
Enter choice: 2

Loaded 50 rows from new_houses.csv
Price #1 Result: 625000.0
Price #2 Result: 580000.0
...
```

### 📊 **Visualization**
```
Available features for X-axis:
1 -> Size(SqFt)
2 -> Bedrooms
3 -> Bathrooms  
4 -> Age(Years)
5 -> DistanceFromCity(KM)
Choose feature number: 1
```
*Generates interactive plot showing Size vs Price with actual/predicted comparison*

---

## 🧠 Technical Deep Dive

### **Automated Preprocessing Pipeline**
- **Data Cleaning**: Removes null values and duplicates
- **Encoding**: `pd.get_dummies()` for categorical variables
- **Scaling**: `StandardScaler` for numerical features
- **Feature Selection**: `SelectKBest` with mutual information scoring

### **Model Arsenal**

#### Classification Models
- `LogisticRegression` - Linear probability modeling
- `RandomForestClassifier` - Ensemble tree-based learning  
- `SVC` - Support Vector Classification
- `BernoulliNB` - Naive Bayes for binary features

#### Regression Models  
- `LinearRegression` - Simple linear relationships
- `RandomForestRegressor` - Non-linear ensemble method
- `SVR` - Support Vector Regression

### **Hyperparameter Optimization**
- **Grid Search**: Exhaustive parameter combinations
- **Cross-Validation**: K-Fold/Stratified validation
- **Adaptive Strategy**: Adjusts based on dataset size
- **Performance Scoring**: Accuracy/R² optimization

### **Smart Data Handling**
- **Dynamic Encoding**: Handles new categorical values in predictions
- **Column Alignment**: Ensures prediction data matches training format
- **Missing Value Imputation**: Automatic handling of incomplete data

---

## 🎯 Use Cases & Examples

### **Real Estate Price Prediction**
- Features: Size, Bedrooms, Location, Age
- Target: House Price
- Type: Regression

### **Customer Churn Prediction** 
- Features: Usage, Payment History, Support Tickets
- Target: Will Churn (Yes/No)
- Type: Binary Classification

### **Product Category Classification**
- Features: Description, Price, Brand, Reviews
- Target: Category (Electronics, Clothing, etc.)
- Type: Multi-class Classification

---

## 🚨 Troubleshooting

### Common Issues

**"Negative R² Score"**
- *Cause*: Dataset too small (< 20 samples)
- *Solution*: Collect more data or use simpler models

**"n_splits cannot be greater than class members"**  
- *Cause*: Small dataset with cross-validation
- *Solution*: Tool automatically adjusts CV strategy

**"Missing columns in prediction file"**
- *Cause*: CSV doesn't match training features
- *Solution*: Ensure all training columns are present

### Performance Tips
- **Minimum Data**: 50+ samples recommended
- **Feature Ratio**: Keep features < samples/5 for best results
- **Categorical Limits**: < 10 categories per feature works best

---

## 🛣️ Roadmap & Future Features

- [ ] **Web Interface** - Flask/Streamlit dashboard
- [ ] **Deep Learning** - TensorFlow/PyTorch integration
- [ ] **Time Series** - ARIMA, Prophet support
- [ ] **Advanced Plots** - SHAP values, feature importance
- [ ] **Model Comparison** - Side-by-side performance metrics
- [ ] **Export Options** - ONNX, Pickle, MLflow integration

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **🐛 Report Bugs** - Open issues with detailed descriptions
2. **💡 Suggest Features** - Share your ideas for improvements  
3. **🔧 Submit PRs** - Fork, improve, and submit pull requests
4. **📖 Improve Docs** - Help make documentation clearer

---

## 🙋‍♂️ Author

**TheGhostLoop** - *First-year CS student passionate about making ML accessible*

Built with:
- ☕ Countless cups of coffee
- 🧠 Curiosity-driven learning  
- 💪 Determination to solve real problems
- 🤝 Community support and feedback

---

## 📄 License

```
MIT License

Copyright (c) 2025 TheGhostLoop

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

⭐ **Star this project if it helped you!** | 🐛 **Report issues** | 🤝 **Contribute**
