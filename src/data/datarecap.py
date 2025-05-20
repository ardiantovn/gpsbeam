import pandas as pd

class DataRecap:

    def __init__(self):
        pass

    def aggregate_metrics(self,
                              df: pd.DataFrame,
                              folder_name: str, 
                              selected_column: str):
        agg_dict = {i: ['mean', 'std'] for i in df.columns if 'acc_percent' in i}
        selected_columns = ['scenario_num', 'num_classes']
        if selected_column not in selected_columns:
            selected_columns.append(selected_column)
        df[selected_column] = df[selected_column].astype(str)
        grouped_df = df.groupby(selected_columns).agg(agg_dict).reset_index()
        
        output_file = f"{folder_name}/grouped_{selected_column}_mean.csv"
        grouped_df.to_csv(output_file, index=False)
        return grouped_df
    
    def read_recap_file(self, fname: str):
        fname = fname.replace('\\', '/')
        df = pd.read_csv(fname)
        return df
