# Machine Learning Assignment

A considerable share of website visitors withhold from engaging in transactions, and the reasons for this can vary. Factors such as a change of intent, competing priorities, or other unforeseen circumstances may lead visitors to avoid completing transactions. While offering visitors the flexibility to navigate the website without committing to transactions is user-friendly, it poses challenges for businesses, potentially impacting revenue. This calls for a Machine Learning-based solution to understand and predict website visitor behavior. Especially, understanding the likelihood of transaction engagement becomes crucial for businesses aiming to optimize their online platforms and enhance the overall user experience.

## Binary Classification Task

In this group assignment (i.e., max 4 members), you are asked to perform a binary classification task. You need to start with reserving 20% of the data for testing purposes.
You will be training 3 models of your choice and comparing their performances. You can choose any two models (Neural Networks, logistic regression, Support Vector Machines (SVM), K-nearest neighbor, etc) for comparison. You will split your training data into training and validation sets to train your models and find the optimal parameters, respectively. You will test the performance of the three models on the same test dataset that you reserved.
You will use:
- training part to train your model,
- validation part to find the optimal parameters of your model,
- test part to evaluate the model with the optimal parameters found on the validation set.

## Results

The exploratory data analysis revealed an imbalanced distribution in the target variable. To address this, a strategic decision was made by the group to implement random oversampling for the underrepresented variable. This approach seeks to correct the imbalance by generating additional instances of the minority class, thereby ensuring a more equitable representation of both classes in the dataset.

<div align = "center">
  
![image](https://github.com/nielsxklesper/Machine_Learning_Challenge/assets/150530277/e2c9952e-e040-4572-95ab-38e484c50cb9)

</div>

The EDA further showed the presence of outliers across multiple features. In response, we opted to employ a function that replaces these outliers with the respective mean of the feature. This decision was driven by the consideration of a quick and efficient implementation, aligning with common practices in data preprocessing. It's important to note, however, that while this method helps in addressing outliers, it comes with a disadvantage: the removal of nuanced information specific to each instance.

```
def replace_outliers(df: pd.DataFrame, inner_fence_multiplier: float = 1.5, outer_fence_multiplier: float = 3.0) -> pd.DataFrame:
    """
    Function used to replace outliers in each continuous column with the mean value.

    Parameters:
        - df (pd.DataFrame): A DataFrame containing the data.
        - inner_fence_multiplier (float): A multiplier to determine the inner fence. Default is 1.5.
        - outer_fence_multiplier (float): A multiplier to determine the outer fence. Default is 3.0.

    Returns:
        - df (pd.DataFrame): A DataFrame with outliers replaced by the mean value.
    """

    for column in ['SystemF2', 'SystemF4', 'Account_Page', 'Account_Page_Time', 'Info_Page', 'Info_Page_Time', 'ProductPage', 'ProductPage_Time', 'GoogleAnalytics_BR', 'GoogleAnalytics_ER', 'GoogleAnalytics_PV']:
        # Calculate the Interquartile Range (IQR)
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        # Calculate the inner and outer fences
        inner_fence_low = Q1 - inner_fence_multiplier * IQR
        inner_fence_high = Q3 + inner_fence_multiplier * IQR
        outer_fence_low = Q1 - outer_fence_multiplier * IQR
        outer_fence_high = Q3 + outer_fence_multiplier * IQR

        # Identify the outliers using the inner and outer fences
        outliers = (df[column] < inner_fence_low) | (df[column] > inner_fence_high) | \
                   (df[column] < outer_fence_low) | (df[column] > outer_fence_high)

        # Replace the outliers with the mean value
        df.loc[outliers, column] = df[column].mean()


    return df
```

The chosen baseline model for this project is the logistic regression classifier. Its selection is based on its attributes of simplicity, interpretability, versatility, and efficiency. These characteristics make it a suitable starting point for the project's modeling efforts, allowing for a clear understanding of the baseline performance before exploring more complex models.

<div align = "center">

| Model                              | Accuracy | Recall | Precision | F1    |
|------------------------------------|----------|--------|-----------|-------|
| Logistic Regression (Oversampled)  | 0.846    | 0.813  | 0.862     | 0.837 |

</div>

In summary, the logistic regression model, after oversampling, demonstrates a good overall performance in terms of accuracy and a balanced trade-off between precision and recall.

The second chosen classifier for this project is the Bagging Classifier, employed as an ensemble learning technique. Ensemble methods aim to improve model performance by combining the predictions of multiple base models. The third model selected is the XGBoost classifier. XGBoost is chosen for its robust predictive capabilities, efficient implementation, and ability to handle complex relationships within the data. It is a gradient-boosting algorithm known for its high performance and versatility, making it suitable for a wide range of machine-learning tasks.

<div align = "center">

| Model                                | Accuracy | Recall | Precision | F1    |
|--------------------------------------|----------|--------|-----------|-------|
| Tuned XGBClassifier (Oversampled)    | 0.941    | 0.985  | 0.902     | 0.942 |
| Tuned BaggingClassifier (Oversampled)| 0.969    | 0.984  | 0.953     | 0.968 |

</div>

Comparing the three models, we observe a clear progression in performance. The baseline logistic regression model, after oversampling, provides a solid starting point. Moving on to the tuned models, both XGBClassifier and BaggingClassifier exhibit substantial improvements across all metrics. However, the tuned BaggingClassifier demonstrates superior performance, achieving the highest accuracy, recall, precision, and F1 score among the three models. This highlights the effectiveness of ensemble methods, especially when fine-tuned, in enhancing predictive capabilities for the given classification task.
