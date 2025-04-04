import pandas as pd

class Load():
    def load(self):    
        # file_path = r"resources\feature_label_chop_range.xlsx"
        file_path = r"resources\feature_label_chop_range 1.csv"
        # df = pd.read_excel(file_path)  
        df = pd.read_csv(file_path)  
        df = df[["Date","lpg_dom"]]
        df.rename(columns={'Date': 'ds', 'lpg_dom': 'y'}, inplace=True)
        df["ds"] = pd.to_datetime(df["ds"])
        df["unique_id"] = 1
        input_data = df
        # print(df.head())

        # input_data = {
        #                 "series": df.to_dict("records")  # Convert DataFrame to list of dictionaries
        #             }
        # print(input_data)
        return input_data