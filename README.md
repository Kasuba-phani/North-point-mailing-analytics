![North Point Analytics Banner](North Point Banner.png)

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

## 🛠️ Methodology & Tech Stack

**Language:** R

**Libraries Used:** `caret`, `dplyr`, `ggplot2`, `randomForest`, `rpart`, `glmnet`, `MASS`

The analysis followed a standard data science pipeline:

1. **Data Preparation:** Cleaning source columns, feature engineering (creating `source_combined`), and categorical factor conversion.
2. **Exploratory Data Analysis (EDA):** Identifying outliers and analyzing distributions of spending.
3. **Modeling:**
* *Classification:* Decision Tree, Logistic Regression, Random Forest.
* *Regression:* Linear Regression, Stepwise Regression (AIC), Random Forest Regression.
* *Clustering:* K-Means (Elbow method for optimal ).


4. **Evaluation:** Confusion Matrices, ROC Curves, MSE (Mean Squared Error), and Lift Charts.

## 📈 Visuals

*(Note: See `figures/` folder for full resolution images)*

> **Variable Importance (Random Forest)**
> *Shows that `Freq` (Frequency) and `Spending` history are the most critical features.*
> > **Decision Tree Logic**
> > *Visual representation of customer purchase rules.*
> 
> 
> ## 🚀 How to Run This Project
> 
> 

1. **Clone the repository**
```bash
git clone [https://github.com/yourusername/north-point-mailing-analytics.git](https://github.com/yourusername/north-point-mailing-analytics.git)

```


2. **Install R Dependencies**
Open RStudio and run:
```r
install.packages(c("dplyr", "ggplot2", "caret", "rpart", "rpart.plot", "Metrics", "randomForest", "glmnet", "MASS"))

```


3. **Data Setup**
* *Note: The dataset `Software_Mailing_List.csv` is not included in this repo for privacy reasons.*
* Place your local copy of the CSV file into the `data/raw/` folder.


4. **Run the Analysis**
* Open `src/analysis_modeling.R`.
* Run the script to generate models and output performance metrics.



## 🔮 Future Improvements

* **Dynamic Optimization:** Build an AI system that updates target lists in real-time based on live campaign feedback.
* **Next-Best-Offer:** Extend models to recommend specific products rather than just binary purchase probability.
* **Automated Dashboarding:** Deploy a Shiny App to monitor ROI and Lift dynamically.

## 👤 Author

**Phanidhar Kasuba**

* Master's in Data Analytics | Webster University
* [LinkedIn Profile](http://linkedin.com/in/phanidhar-kasuba) | [Email](mailto:pkasubavenkatana@webster.edu)

---

*This project was completed as part of the CSDA 6010 Analytics Practicum.*

```
