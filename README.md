#Using Machine Learning Models for Predicting Energy Billing Amounts Using Consumer Data

**Columms and their description:**

1. **customer_id**: Unique identifier for each customer.
2. **region**: Geographic region of the customer.
3. **energy_consumption_kwh**: Total energy consumption in kilowatt-hours.
4. **peak_hours_usage**: Energy usage during peak hours.
5. **off_peak_usage**: Energy usage during off-peak hours.
6. **renewable_energy_pct**: Percentage of energy from renewable sources.
7. **billing_amount**: Total billing amount.
8. **household_size**: Number of people in the household.
9. **temperature_avg**: Average temperature.
10. **income_bracket**: Income bracket of the household.
11. **smart_meter_installed**: Whether a smart meter is installed.
12. **time_of_day_pricing**: Whether time-of-day pricing is used.
13. **annual_energy_trend**: Annual trend in energy consumption.
14. **solar_panel**: Whether solar panels are installed.
15. **target_high_usage**: Whether the household is targeted for high usage.

**The document is structured as follows:**

1. **Data Preparation:** Details the steps taken to load, preprocess, and split the dataset.
2. **Model Training and Evaluation:** Describes the process of training each model and evaluating their performance using the test data.
3. **Results and Analysis:** Presents the evaluation metrics for each model and identifies the best and worst performing models.
4. **Conclusion:** Summarizes the findings and provides recommendations based on the analysis.



**Data Preparation:**

The dataset is loaded using pd.read_csv.
The features (X) and target (y) are defined.
The dataset is split into training and testing sets using train_test_split.

**Model training:**

Three different models are trained:
1. Linear Regression
2. Random Forest Regressor
3. Decision Tree Regressor

**Best and Worst Models:**

The best and worst models are determined based on the R-squared value. The model with the highest R-squared value is considered the best, and the model with the lowest R-squared value is considered the worst.
