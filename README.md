# Fraud Detection Using Neural Network (Credit Card Dataset)

## 📁 Dataset Source

[Credit Card Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

* **Instances:** 284,807 transactions
* **Fraudulent Transactions:** 492 (\~0.17%)
* **Features:** 30 (28 anonymized via PCA + Time + Amount)
* **Target:** `Class` (1 = Fraud, 0 = Non-Fraud)

---

## 🧠 Objective

Build a robust, explainable, and interactive neural network-based model to detect fraudulent transactions using:

* Data normalization
* Undersampling to balance class imbalance
* Neural networks (shallow and deep)
* Model evaluation (ROC, PR AUC, etc.)
* Shiny dashboard for CSV upload and real-time fraud scoring

---

## 🧼 Data Preprocessing Steps

1. **Normalization**

   * Scaled `Time` and `Amount` columns to range \[0, 1] to ensure fair learning by the neural network.

2. **Handling Imbalance**

   * Applied random undersampling to create a balanced dataset:

     * 492 fraud cases
     * 492 randomly selected non-fraud cases

3. **Train-Test Split**

   * 70% Training, 30% Testing
   * Stratified sampling to maintain class balance

4. **Feature Selection**

   * Used all features except `Class` as inputs

---

## 🧠 Model Development

### 1. **Shallow Neural Network**

* Architecture: `Input -> 5 Hidden Neurons -> Output`
* Tool: `neuralnet` package in R

### 2. **Deep Neural Network**

* Architecture: `Input -> 10 Hidden Neurons -> 5 Hidden Neurons -> Output`
* Trained in parallel for comparison

---

## 📊 Model Evaluation Metrics

| Metric      | Value   | Interpretation                           |
| ----------- | ------- | ---------------------------------------- |
| Accuracy    | \~94.2% | Correct predictions across all classes   |
| Kappa       | \~0.88  | Agreement corrected for chance           |
| Sensitivity | \~91.2% | Fraud cases correctly detected           |
| Specificity | \~97.3% | Legitimate transactions correctly passed |
| ROC AUC     | 0.976   | Excellent ability to separate classes    |
| PR AUC      | 0.981   | High precision-recall performance        |

---

## 📊 Visualizations

* ROC Curve
* Precision-Recall Curve
* Neural Network Diagram (Shallow and Deep)

---

## 💻 Shiny Dashboard

**Features:**

* Interactive dashboard for exploration and scoring
* CSV upload interface for fraud prediction
* Model visualizations
* Metrics and performance summary

**Limitations:**

* Not designed for high-frequency real-time detection
* Needs hosted deployment (e.g., shinyapps.io) for production use

---

## 🧾 How to Use

1. Clone repository and open RStudio
2. Run the training script (includes model + dashboard)
3. Launch `shinyApp(ui, server)` to access dashboard
4. Upload new CSV data (same format, no 'Class')
5. View fraud predictions and probabilities instantly

---

## 📌 Summary

This project demonstrates how a neural network, even on a highly imbalanced dataset, can achieve strong performance with proper preprocessing and evaluation. Combined with a Shiny dashboard, the model becomes interactive and practical for real-world fraud detection workflows.

---

## 👨‍💻 Built With

* **R** (neuralnet, caret, dplyr, shiny, pROC, PRROC)
* **Dataset**: PCA-anonymized credit card transactions (Kaggle)
* **Deployment**: Shiny (Local or cloud-hosted)

---

## 📬 Contact

*This project was built for educational, analytical, and demonstration purposes. For questions, contact the creator or submit a pull request.*
