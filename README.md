Here’s a tailored `README.md` for your new version of **MLCrafter**:

---

# 🤖 MLCrafter v2

**MLCrafter v2** is an enhanced terminal-based tool built to help you create, train, and interact with custom **ML models** — both **Classification** and **Regression** — from your own CSV datasets. It's built using `scikit-learn`, supports preprocessing, intelligent feature selection, model evaluation, and even lets you load saved models to make live predictions.

---

## ✨ Features

* 📁 Load any CSV dataset
* 🧠 Auto-detect **problem type** (Classification or Regression)
* 🔢 Handles numeric and categorical columns automatically
* 📈 Feature selection using `SelectKBest`
* 🧪 Trains multiple models with `GridSearchCV`
* 🏆 Auto-picks the **best model**
* 💾 Save and reuse models with metadata
* 🔮 Predict using live user input (even with strings!)
* 📊 Visualize actual vs predicted graph (regression only)

---

## 📂 Folder Structure

```
MLCrafter-v2/
├── savedmodels/          # Trained and saved ML models
├── csvfiles/       # Sample CSVs to test
├── main.py               # Main script (Run this)
├── README.md             # You're here
└── requirements.txt      # Required libraries
```

---

## ⚙️ Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
pandas
numpy
matplotlib
scikit-learn
joblib
```

---

## 🚀 How to Run

```bash
python main.py
```

You'll be prompted with:

```
1 -> Train a new model
2 -> Use trained model
3 -> Exit
```

---

## 🔄 Workflow

1. **Train a Model**

   * Load a CSV dataset
   * Choose target column
   * Tool detects if it's classification/regression
   * Data is encoded, filtered, and preprocessed
   * Multiple models are trained using `GridSearchCV`
   * Best model is selected and saved

2. **Use Existing Model**

   * Load any previously trained model
   * Enter input data dynamically
   * Tool handles missing dummy variables internally
   * Predictions printed directly to terminal

3. **Visualization (Optional)**

   * For regression models, you can generate an **Actual vs Predicted** plot

---

## 🧠 Internals

* Feature selection: `SelectKBest` using `mutual_info_classif` or `mutual_info_regression`
* Models used:

  * Classification: `LogisticRegression`, `RandomForest`, `SVC`, `BernoulliNB`
  * Regression: `LinearRegression`, `RandomForestRegressor`, `SVR`
* Saves models using `joblib`, preserving pipelines and scalers
* Handles dynamic prediction input with `get_dummies` mapping and column alignment

---

## 🙋‍♂️ Author

Built with 💻 and frustration-fueled learning by **TheGhostLoop**.

---

## 📜 License

MIT License — free to use, modify, and share.

---

Let me know when you're ready — I can help you write `requirements.txt`, push the code to GitHub, and add a preview badge too.
