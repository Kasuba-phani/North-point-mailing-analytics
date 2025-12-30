![North Point Analytics Banner](project_banner.png)

# 📬 North-Point Software Mailing Analytics

![R](https://img.shields.io/badge/Language-R-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Domain](https://img.shields.io/badge/Domain-Marketing%20Analytics-orange)

## 📖 Executive Summary
**Business Problem:** North-Point Software has a database of 5 million potential customers but faces high costs and low response rates (5.3%) with random mailing. The goal is to select the top 200,000 prospects most likely to purchase.

**Solution:** This project utilizes predictive modeling (Classification & Regression) and Customer Segmentation (Clustering) to optimize the mailing list. By targeting high-probability buyers, we can significantly reduce marketing waste and increase ROI.

## 📊 Key Findings & Results

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Best Classifier** | **Random Forest** | Achieved **81.7% accuracy** in predicting purchasers. |
| **Lift** | **1.9x** | Targeting the top decile yields nearly double the response rate of random selection. |
| **Cost Savings** | **70% Reduction** | Can reduce mail volume by 70% while retaining the majority of sales. |
| **Key Drivers** | **Frequency & Recency** | Transaction history and web interaction were the strongest predictors of spending. |

### 🔍 Customer Segments (Clustering)
Using K-Means clustering, the customer base was segmented into three distinct tiers:
1.  **VIPs (Cluster 3):** High frequency, high spend. Strategy: Loyalty rewards.
2.  **Warm Prospects:** Moderate frequency. Strategy: Re-engagement offers.
3.  **Cold Leads:** Low activity. Strategy: Awareness campaigns or exclusion to save costs.

## 📂 Repository Files

* `Phani_Project2_Final.R` - **Main Analysis Script**: Contains data cleaning, modeling (Regression/Classification), and evaluation code.
* `Software_Mailing_List.csv` - **Dataset**: The historical customer data used for training the models.
* `Phanindhar_Kasuba_Project_2_Final.docx` - **Full Report**: detailed explanation of methodology and business recommendations.
* `A4697EF6-4961-44C1-A101-4CCFD56DA849.pdf` - **Presentation**: Executive slide deck summarizing the project.

## 🛠️ Methodology & Tech Stack

**Language:** R  
**Libraries Used:** `caret`, `dplyr`, `ggplot2`, `randomForest`, `rpart`, `glmnet`, `MASS`

The analysis followed a standard data science pipeline:
1.  **Data Preparation:** Cleaning source columns, feature engineering (creating `source_combined`), and categorical factor conversion.
2.  **Exploratory Data Analysis (EDA):** Identifying outliers and analyzing distributions of spending.
3.  **Modeling:**
    * *Classification:* Decision Tree, Logistic Regression, Random Forest.
    * *Regression:* Linear Regression, Stepwise Regression (AIC), Random Forest Regression.
    * *Clustering:* K-Means (Elbow method for optimal $k$).
4.  **Evaluation:** Confusion Matrices, ROC Curves, MSE (Mean Squared Error), and Lift Charts.

## 🚀 How to Run This Project

1.  **Clone the repository**
2.  **Install R Dependencies**
    ```r
    install.packages(c("dplyr", "ggplot2", "caret", "rpart", "rpart.plot", "Metrics", "randomForest", "glmnet", "MASS"))
    ```
3.  **Run the Analysis**
    * Open `Phani_Project2_Final.R` in RStudio.
    * Ensure `Software_Mailing_List.csv` is in your working directory.
    * Run the script to generate models and output performance metrics.

## 👤 Author
**Phanidhar Kasuba**
* Master's in Data Analytics | Webster University

---
*This project was completed as part of the CSDA 6010 Analytics Practicum.*
