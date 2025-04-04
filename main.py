from backend.components.forecast import Forecast
from backend.components.loader import Load

import matplotlib.pyplot as plt
import numpy as np

# input_data = Load().load()
# forecast_df = Forecast().forecast(input_data)

# print(forecast_df.head())

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def MAE(y_true, y_pred):
    N = len(y_true)
    return (1/N)*np.sum(abs(y_true - y_pred))

def MSE(y_true, y_pred):
    N = len(y_true)
    return (1/N)*np.sum((y_true - y_pred)**2)

def main():
    input_data = Load().load()
    #1st : 176 best result is 20% so 36
    #2nd : 813
    # train_data = input_data[:144]
    # test_data = input_data[144:]
    train_data = input_data[:763]
    test_data = input_data[763:]

    forecast_train = Forecast().forecast(train_data)
    forecast_df = Forecast().forecast(input_data)


    # print(input_data.head())
    # print(forecast_df.head())
    print("Input data: ", len(input_data))
    print("Forecast train: ", len(train_data))
    print("Forecast test: ", len(test_data))
    print("Forecast df: ", len(forecast_df))

    ### Plotting ###

    y = input_data['y'].values
    y_train = train_data['y'].values
    y_test = test_data['y'].values

    predictions = forecast_train['timesfm'].values
    combined_values = np.concatenate([y_train, predictions])
    time_steps = np.arange(len(combined_values))
    
    #R2
    r2 = r_squared(y_test, predictions)
    print(f"R²: {r2}")
    
    #MAE
    mae = MAE(y_test, predictions)
    print(f"MAE: {mae}")
    
    #MSE
    mse = MSE(y_test, predictions)
    print(f"MSE: {mse}")

    plt.figure(figsize=(10, 6))

    plt.plot(time_steps[:len(y)], y, label='Actual Values', color='blue', alpha=0.7)
    plt.plot(time_steps[len(y_train):], predictions, label='Predicted Values', color='red', linestyle='--', alpha=0.7)

    plt.xlabel('Time Steps')
    plt.ylabel('lpg_dom')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.grid(True)

    plt.show()

    # Scatter plot comparing predicted values directly against actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, label='Predicted vs Actual', color='green', alpha=0.7)
    min_val = min(min(y_test), min(predictions))
    max_val = max(max(y_test), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
