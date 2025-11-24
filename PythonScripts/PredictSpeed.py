import os
import shap
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_SAVED_FOLDER = os.path.join(SCRIPT_DIR, "..", "CreatedDataset")

# ------ Customize Parameters ------
DATASET_FILE_NAME = "SpeedPredictorDataset_1124_2339.csv"
bDisplayImportanceOfX = False
bDisplayErrorDistribution = True
#-----------------------------------

# ------ Machine Learning ------
def GetPred_LinearRegression(X_Train, Y_Train, X_Test):
    Model_LR = LinearRegression()
    Model_LR.fit(X_Train, Y_Train)
    PredictedVals_LR = Model_LR.predict(X_Test)
    return PredictedVals_LR

def GetPred_SVR(X_Train, Y_Train, X_Test):
    Model_SVR = SVR(kernel="rbf")
    Model_SVR.fit(X_Train, Y_Train)
    PredictedVals_SVR = Model_SVR.predict(X_Test)
    return PredictedVals_SVR

def GetPred_XGBoost(X_Train, Y_Train, X_Test):
    Model_XGB = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    objective="reg:squarederror",
    random_state=42
    )
    Model_XGB.fit(X_Train, Y_Train)
    PredictedVals_XGB = Model_XGB.predict(X_Test)
    return PredictedVals_XGB
# ------------------------------

# ------ Evaluation ------
def GetEvaluation(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    me   = np.mean(y_pred - y_true)
    return round(mae, 2), round(rmse, 2), round(r2, 2), round(me, 2)
# ------------------------

def main():

    DataFrame = pd.read_csv(os.path.join(OUTPUT_SAVED_FOLDER, DATASET_FILE_NAME))

    # Explanatory variable X : Response variable Y = 8 : 2
    X = DataFrame[['accel_STD','accel_RMS','accel_DomFreq1','accel_DomFreq2','ang_STD','ang_RMS','ang_DomFreq1','ang_DomFreq2']]
    Y = DataFrame['TargetSpeed']
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(
        X, Y, test_size=0.2, shuffle=True, random_state=20
    )

    # Predict by Machine Learning
    PredictedVals_LR = GetPred_LinearRegression(X_Train, Y_Train, X_Test)
    PredictedVals_SVR = GetPred_SVR(X_Train, Y_Train, X_Test)
    PredictedVals_XGB = GetPred_XGBoost(X_Train, Y_Train, X_Test)
    PredictedVals_Ensemble = (PredictedVals_LR + PredictedVals_SVR + PredictedVals_XGB) / 3.0

    print("===== Regression Evaluation =====")
    mae, rmse, r2, me = GetEvaluation(Y_Test, PredictedVals_LR)
    print(f"LinearRegression | (MAE, RMSE, R2, ME) = ({mae}, {rmse}, {r2}, {me})")
    mae, rmse, r2, me = GetEvaluation(Y_Test, PredictedVals_SVR)
    print(f"SVR              | (MAE, RMSE, R2, ME) = ({mae}, {rmse}, {r2}, {me})")
    mae, rmse, r2, me = GetEvaluation(Y_Test, PredictedVals_XGB)
    print(f"XGBoost          | (MAE, RMSE, R2, ME) = ({mae}, {rmse}, {r2}, {me})")
    mae, rmse, r2, me = GetEvaluation(Y_Test, PredictedVals_Ensemble)
    print(f"Ensemble         | (MAE, RMSE, R2, ME) = ({mae}, {rmse}, {r2}, {me})")

    # Error Distribution
    if bDisplayErrorDistribution:
        residuals = PredictedVals_XGB - Y_Test

        plt.figure(figsize=(8,6))
        plt.hist(residuals, bins=30, color='skyblue', edgecolor='black')
        plt.xlabel("Prediction Error (Predicted - True)")
        plt.ylabel("Count")
        plt.title("XGBoost Residuals Histogram")
        plt.show()

        plt.figure(figsize=(8,6))
        plt.scatter(Y_Test, PredictedVals_XGB, alpha=0.5)
        plt.plot([Y_Test.min(), Y_Test.max()], [Y_Test.min(), Y_Test.max()], 'r--')  # 完全一致線
        plt.xlabel("True Speed (km/h)")
        plt.ylabel("Predicted Speed (km/h)")
        plt.title("XGBoost Predicted vs True")
        plt.show()

        plt.figure(figsize=(6,6))
        plt.boxplot(residuals, vert=True)
        plt.ylabel("Prediction Error")
        plt.title("XGBoost Residuals Boxplot")
        plt.show()


    # SHAP Importance
    if bDisplayImportanceOfX:
        feature_names = X.columns

        Model_XGB_imp = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            objective="reg:squarederror",
            random_state=42
        )
        Model_XGB_imp.fit(X_Train, Y_Train)

        explainer = shap.TreeExplainer(Model_XGB_imp)
        shap_values = explainer.shap_values(X_Test)

        shap.summary_plot(shap_values, X_Test, feature_names=feature_names, plot_type="bar")
        shap.summary_plot(shap_values, X_Test, feature_names=feature_names)

if __name__ == "__main__":
    main()
