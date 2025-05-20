import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

class DataPlot:
    def __init__(self):
        self.figsize = (8, 6)

    def plot_loss_vs_epochs(self, 
                            df: pd.DataFrame, 
                            folder_name: str, 
                            format='pdf',
                            is_finetune: bool=False,
                            fontsize: int = 12,
                            axis_fontsize: int = 12,
                            tick_fontsize: int = 12,
                            legend_fontsize: int = 12):
        """
        Plot training and validation average loss 
        versus epochs from the CSV data.
        
        Args:
            df (pd.DataFrame): DataFrame containing loss data
            folder_name (str): Path to save output plot
            format (str): Output format ('pdf' or other)
            is_finetune (bool): Whether this is for a finetuned model
            fontsize (int): Font size for plot text elements (default: 12)
            axis_fontsize (int): Font size for axis labels (default: 12) 
            tick_fontsize (int): Font size for tick labels (default: 12)
            legend_fontsize (int): Font size for legend (default: 12)
        """
        # Set font sizes
        plt.rc('font', size=fontsize)          
        plt.rc('axes', titlesize=fontsize)    
        plt.rc('axes', labelsize=axis_fontsize)    
        plt.rc('xtick', labelsize=tick_fontsize)    
        plt.rc('ytick', labelsize=tick_fontsize)    
        plt.rc('legend', fontsize=legend_fontsize)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        epochs = df['epoch']
        train_loss = df['train_avg_loss']
        val_loss = df['val_avg_loss']

        ax.plot(epochs, train_loss, label='Training Loss', marker='o')
        ax.plot(epochs, val_loss, label='Validation Loss', marker='s')
        # Add horizontal line for train entropy
        if 'train_entropy' in df.columns and len(df['train_entropy']) > 0:
            train_entropy = df['train_entropy'].iloc[0]  # Get the first value
            ax.axhline(y=train_entropy, color='r', linestyle='--', 
                      label=f'Train Data Set Entropy: {train_entropy:.2f} nats')        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Loss (nats)')
        if format != 'pdf':
            if is_finetune:
                ax.set_title('Training and Test Loss vs Epochs (on finetuned model)')
            else:
                ax.set_title('Training and Test Loss vs Epochs')
        ax.legend()
        ax.grid(True)
         # Ensure x-axis ticks are integers
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # plt.tight_layout()
        if is_finetune:
            filename = f"{folder_name}/finetune_train_val_loss_vs_epochs.pdf"
        else:
            filename = f"{folder_name}/train_val_loss_vs_epochs.pdf"
        fig.savefig(filename, format=format)
        plt.close()
        print(f"Saved plot to {filename}")
        
    
    def plot_overhead_savings(self,
                              df: pd.DataFrame, 
                              folder_name: str,
                              format='pdf'):
        """
        Plot overhead savings vs confidence level for different prediction horizons.
        
        Args:
            df (pd.DataFrame): DataFrame containing overhead savings data
            save_path (str, optional): Path to save the plot as PDF. If None, plot is displayed but not saved.
        """
        # Create a figure for the plot
        plt.figure(figsize=(10, 6))
        
        # Find all overhead saving columns
        overhead_cols = [col for col in df.columns if 'overhead' in col.lower()]
        
        # Extract confidence levels from column names
        confidence_levels = sorted(list(set([
            int(col.split('_')[-2].replace('pct', '')) 
            for col in overhead_cols if 'pct' in col
        ])))
        
        # Determine prediction horizons available
        pred_indices = sorted(list(set([
            int(col.split('_')[0].replace('pred', '')) 
            for col in overhead_cols if col.startswith('pred')
        ])))
        
        # Colors for different prediction horizons (extend if needed)
        colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray']
        labels = [f'{"Current Beam" if i==0 else f"Future Beam-{i}"} Top1 Prediction' for i in pred_indices]
        
        # Plot overhead savings vs confidence level for each prediction horizon
        for i, pred_idx in enumerate(pred_indices):
            overhead_values = []
            for conf in confidence_levels:
                col_name = f'pred{pred_idx}_top1_overhead_saving_for_{conf}pct_conf'
                if col_name in df.columns:
                    overhead_values.append(df[col_name].values[0])
            
            if overhead_values:  # Only plot if we have data
                plt.plot(confidence_levels, overhead_values, marker='o', 
                        color=colors[i % len(colors)], label=labels[i], linewidth=2)
        
        plt.xlabel('Guaranteed Reliability [1 - Outage Probability] (%)')
        plt.ylabel('Overhead Savings (%)')
        if format != 'pdf':
            plt.title('Overhead Savings vs Guaranteed Reliability for Different Prediction Steps')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # Set x-axis to only show the specific confidence levels
        plt.xticks(confidence_levels)
        
        # Save to PDF if save_path is provided
        if folder_name:
            filename = f"{folder_name}/overhead_savings_vs_confidence_level.pdf"
            print(f"Saving plot to {filename}")
            plt.savefig(filename, format=format, bbox_inches='tight')
    
    def plot_reliability_metrics(self, df: pd.DataFrame, folder_path: str, format='pdf', y_max=100, fontsize: int = 12):
        """
        Plot reliability metrics for power loss across prediction steps.
        
        Args:
            df (pd.DataFrame): DataFrame containing metrics
            folder_path (str): Path to save output plot
            format (str): Output format ('pdf' or other)
            y_max (float): Maximum value for y-axis (default: 100)
            fontsize (int): Font size for plot text elements (default: 12)
        """
        # Extract reliability metrics for each prediction step
        reliability_3db = []
        reliability_1db = []
        for i in range(4):
            col_3db = f'pred{i}_top1_reliability_power_loss_leq_3db_pct'
            col_1db = f'pred{i}_top1_reliability_power_loss_leq_1db_pct'
            reliability_3db.append(df[col_3db].values[0])
            reliability_1db.append(df[col_1db].values[0])

        # Create bar plot
        plt.figure(figsize=(10,5))
        x = range(4)
        width = 0.35
        step_names = ['Current beam \n Top-1 Prediction', 
                    'Future beam1 \n Top-1 Prediction', 
                    'Future beam2 \n Top-1 Prediction', 
                    'Future beam3 \n Top-1 Prediction']

        # Set font sizes
        plt.rc('font', size=fontsize)          
        plt.rc('axes', titlesize=fontsize)    
        plt.rc('axes', labelsize=fontsize)    
        plt.rc('xtick', labelsize=fontsize)    
        plt.rc('ytick', labelsize=fontsize)    
        plt.rc('legend', fontsize=fontsize)    
        plt.rc('figure', titlesize=fontsize)  

        # Create bars - switched order so ≤1dB appears first
        bars1 = plt.bar([i - width/2 for i in x], reliability_1db, width, label='≤ 1dB', color='#155E95')
        bars2 = plt.bar([i + width/2 for i in x], reliability_3db, width, label='≤ 3dB', color='#F93827')

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom')

        plt.xlabel('Model Output')
        plt.ylabel('Reliability %')
        if format != 'pdf':
            plt.title('Mean Power Loss Reliability')
        plt.xticks(x, step_names)
        plt.ylim(bottom=92, top=y_max)
        plt.legend()
        
        # Adjust layout to prevent x-axis label cropping
        plt.tight_layout()
        
        # Save plot
        print(f"Saving plot to {folder_path}/reliability_metrics.{format}")
        plt.savefig(f"{folder_path}/reliability_metrics.{format}", format=format, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_overhead_savings_and_reliability_metrics(self, 
                                                      df: pd.DataFrame, 
                                                      folder_path: str, 
                                                      format='pdf', 
                                                      orientation='horizontal',
                                                      fontsize: int = 12,
                                                      axis_fontsize: int = 12,
                                                      bar_fontsize: int = 12,
                                                      legend_fontsize: int = 12,
                                                      y_max: float = 100):
        """
        Plot overhead savings and reliability metrics side by side.
        
        Args:
            df (pd.DataFrame): DataFrame containing metrics
            folder_path (str): Path to save output plot
            format (str): Output format ('pdf' or other)
            orientation (str): 'horizontal' or 'vertical' layout
            fontsize (int): Font size for plot text elements (default: 12)
            axis_fontsize (int): Font size for axis labels (default: 12)
            bar_fontsize (int): Font size for bar value labels (default: 12)
            legend_fontsize (int): Font size for legend (default: 12)
        """
        if orientation == 'horizontal':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            
        # Set font sizes
        plt.rc('font', size=fontsize)          
        plt.rc('axes', titlesize=fontsize)    
        plt.rc('axes', labelsize=axis_fontsize)    
        plt.rc('xtick', labelsize=axis_fontsize)    
        plt.rc('ytick', labelsize=axis_fontsize)    
        plt.rc('legend', fontsize=legend_fontsize)    
        plt.rc('figure', titlesize=fontsize)
            
        # Plot 1: Overhead Savings
        overhead_cols = [col for col in df.columns if 'overhead' in col.lower()]
        confidence_levels = sorted(list(set([
            int(col.split('_')[-2].replace('pct', '')) 
            for col in overhead_cols if 'pct' in col
        ])))
        pred_indices = sorted(list(set([
            int(col.split('_')[0].replace('pred', '')) 
            for col in overhead_cols if col.startswith('pred')
        ])))
        
        colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray']
        labels = [f'{"Current Beam" if i==0 else f"Future Beam-{i}"} Top1 Prediction' for i in pred_indices]
        
        ax1.text(0.5, 1.1, '(a) Overhead Savings', 
                horizontalalignment='center', transform=ax1.transAxes, fontsize=axis_fontsize)
        
        # Create DataFrame for overhead savings plot
        overhead_data = pd.DataFrame()
        
        for i, pred_idx in enumerate(pred_indices):
            overhead_values = []
            for conf in confidence_levels:
                col_name = f'pred{pred_idx}_top1_overhead_saving_for_{conf}pct_conf'
                if col_name in df.columns:
                    value = df[col_name].values[0]
                    overhead_values.append(value)
                    
                    # Add to plot data DataFrame
                    overhead_data = pd.concat([overhead_data, pd.DataFrame({
                        'prediction': [labels[i]],
                        'confidence_level': [conf],
                        'overhead_saving': [value]
                    })], ignore_index=True)
            
            if overhead_values:
                ax1.plot(confidence_levels, overhead_values, marker='o', 
                        color=colors[i % len(colors)], label=labels[i], linewidth=2)
        
        ax1.set_xlabel('Guaranteed Reliability [1 - Outage Probability] (%)')
        ax1.set_ylabel('Overhead Savings (%)')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        ax1.set_xticks(confidence_levels)
        
        # Save overhead savings data
        overhead_csv_filename = f"{folder_path}/overhead_savings_data.csv"
        overhead_data.to_csv(overhead_csv_filename, index=False)
        print(f"Saving overhead savings data to {overhead_csv_filename}")
        
        # Plot 2: Reliability Metrics
        reliability_3db = []
        reliability_1db = []
        
        # Create DataFrame for reliability metrics plot
        reliability_data = pd.DataFrame()
        
        for i in range(4):
            col_3db = f'pred{i}_top1_reliability_power_loss_leq_3db_pct'
            col_1db = f'pred{i}_top1_reliability_power_loss_leq_1db_pct'
            value_3db = df[col_3db].values[0]
            value_1db = df[col_1db].values[0]
            reliability_3db.append(value_3db)
            reliability_1db.append(value_1db)
            
            step_name = f'{"Current beam" if i==0 else f"Future beam{i}"} Top-1 Prediction'
            
            # Add to reliability data DataFrame
            reliability_data = pd.concat([reliability_data, pd.DataFrame({
                'prediction': [step_name],
                'threshold': ['≤ 1dB'],
                'reliability_percent': [value_1db]
            })], ignore_index=True)
            
            reliability_data = pd.concat([reliability_data, pd.DataFrame({
                'prediction': [step_name],
                'threshold': ['≤ 3dB'],
                'reliability_percent': [value_3db]
            })], ignore_index=True)

        x = range(4)
        width = 0.35
        step_names = ['Current beam \n Top-1 Prediction', 
                    'Future beam1 \n Top-1 Prediction', 
                    'Future beam2 \n Top-1 Prediction', 
                    'Future beam3 \n Top-1 Prediction']

        ax2.text(0.5, 1.1, '(b) Mean Power Loss Reliability', 
                horizontalalignment='center', transform=ax2.transAxes, fontsize=axis_fontsize)

        bars1 = ax2.bar([i - width/2 for i in x], reliability_1db, width, label='≤ 1dB', color='#6A80B9')
        bars2 = ax2.bar([i + width/2 for i in x], reliability_3db, width, label='≤ 3dB', color='#F6C794')

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=bar_fontsize)

        ax2.set_xlabel('Model Output')
        ax2.set_ylabel('Reliability %')
        ax2.set_xticks(x, step_names)
        ax2.set_ylim(bottom=92, top=y_max)
        ax2.legend()

        plt.tight_layout()
        
        # Save reliability data
        reliability_csv_filename = f"{folder_path}/reliability_metrics_data.csv"
        reliability_data.to_csv(reliability_csv_filename, index=False)
        print(f"Saving reliability metrics data to {reliability_csv_filename}")
        
        # Save combined data for reference
        combined_data = pd.DataFrame()
        
        # Add overhead savings data
        for i, pred_idx in enumerate(pred_indices):
            for conf in confidence_levels:
                col_name = f'pred{pred_idx}_top1_overhead_saving_for_{conf}pct_conf'
                if col_name in df.columns:
                    combined_data.loc[f'overhead_pred{pred_idx}_{conf}pct', 'value'] = df[col_name].values[0]
                    combined_data.loc[f'overhead_pred{pred_idx}_{conf}pct', 'prediction'] = labels[i]
                    combined_data.loc[f'overhead_pred{pred_idx}_{conf}pct', 'confidence'] = conf
        
        # Add reliability data
        for i in range(4):
            combined_data.loc[f'reliability_1db_pred{i}', 'value'] = reliability_1db[i]
            combined_data.loc[f'reliability_1db_pred{i}', 'prediction'] = step_names[i].replace('\n', ' ')
            combined_data.loc[f'reliability_1db_pred{i}', 'threshold'] = '1dB'
            
            combined_data.loc[f'reliability_3db_pred{i}', 'value'] = reliability_3db[i]
            combined_data.loc[f'reliability_3db_pred{i}', 'prediction'] = step_names[i].replace('\n', ' ')
            combined_data.loc[f'reliability_3db_pred{i}', 'threshold'] = '3dB'
        
        # Save combined plot data
        combined_csv_filename = f"{folder_path}/combined_metrics_data.csv"
        combined_data.to_csv(combined_csv_filename)
        print(f"Saving combined plot data to {combined_csv_filename}")
        
        # Save combined plot
        filename = f"{folder_path}/combined_metrics_{orientation}.{format}"
        print(f"Saving combined plot to {filename}")
        plt.savefig(filename, format=format, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_boxplot(self, 
                 df: pd.DataFrame, 
                 folder_name: str,
                 selected_column: str, 
                 num_classes: int,
                 metric: str='pred0_top1_test_acc_percent'):
        """
        Create and save a boxplot to compare a metric across different values of a selected column,
        with colors differentiated by scenario_num.
        
        Args:
        df (pd.DataFrame): The DataFrame containing the data
        folder_name (str): The name of the folder to save the plot
        selected_column (str): The column to use for grouping (e.g., 'seednum', 'splitting_method')
        metric (str): The metric to plot (default is 'pred0_top1_test_acc_percent')
        """
        df[selected_column] = df[selected_column].astype(str)
        plt.figure(figsize=(14, 8))  # Increased figure size to accommodate legend
        
        # Create the boxplot with hue for scenario_num
        sns.boxplot(x=selected_column, y=metric, hue='scenario_num', data=df)
        
        plt.title(f'{metric} Distribution by {selected_column} and Scenario')
        plt.xlabel(selected_column)
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        
        # Adjust legend
        plt.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save the plot as PDF
        filename = f"{folder_name}/boxplot_{selected_column}_{metric}_by_scenario_num_classes_{num_classes}.pdf"
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close()  # Close the figure to free up memory

    def plot_label_distribution(self, 
                                dataset_dict: dict, 
                                label_column_name: str, 
                                folder_name: str,
                                scenario_num_str: str,
                                splitting_method: str,
                                fontsize: int=12,
                                margin_factor: float=1.2):
        """
        Plot bar chart of label distribution for train and test datasets.

        Args:
        dataset_dict (dict): Dictionary containing 'train' and 'test' datasets
        label_column_name (str): Name of the column containing labels
        folder_name (str): The name of the folder to save the plot
        max_y (float, optional): Maximum value for y-axis. If None, automatically determined.
        """
        # Create figure with larger height to accommodate labels
        plt.figure(figsize=(16, 10))  # Increased height from 8 to 10

        # Prepare data for plotting
        if 'val' in dataset_dict:
            datasets = ['train', 'val', 'test']
        else:
            datasets = ['train', 'test']
        label_counts = {}

        max_count = 0
        for dataset in datasets:
            if dataset in dataset_dict:
                labels = dataset_dict[dataset][label_column_name]
                unique_labels, counts = np.unique(labels, return_counts=True)
                label_counts[dataset] = {label: count for label, count in zip(unique_labels, counts)}
                max_count = max(max_count, max(counts))

        # Get all unique labels across datasets
        all_labels = sorted(set().union(*[set(counts.keys()) for counts in label_counts.values()]))

        # Prepare data for grouped bar chart
        x = np.arange(len(all_labels))
        if len(datasets) == 2:
            width = 0.35
        else:
            width = 0.2
        
        # Create a dataframe to store the plot data
        plot_data = []
        for dataset in datasets:
            if dataset in label_counts:
                for label in all_labels:
                    count = label_counts[dataset].get(label, 0)
                    plot_data.append({
                        'dataset': dataset,
                        'label': label,
                        'count': count
                    })
        plot_df = pd.DataFrame(plot_data)
        
        # Save the plot data to CSV
        csv_filename = f"{folder_name}/label_distribution_data_scenario_{scenario_num_str}_sampling_{splitting_method}.csv"
        print(f"Saving plot data to {csv_filename}")
        plot_df.to_csv(csv_filename, index=False)
        
        # Plot bars for each dataset
        for i, dataset in enumerate(datasets):
            if dataset in label_counts:
                counts = [label_counts[dataset].get(label, 0) for label in all_labels]
                bars = plt.bar(x + (i-0.5)*width, counts, width, label=dataset.capitalize())
                
                # Add number on top of each bar with smaller font
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{count}',
                            ha='center', va='bottom', fontsize=fontsize-4, rotation=90)  # Reduced font size

        plt.xlabel('Labels', fontsize=fontsize)
        plt.ylabel('Sample Count', fontsize=fontsize)
        plt.title(f'Scenario {scenario_num_str} with {splitting_method} sampling method:\nLabel Distribution in Train and Test Sets', 
                 fontsize=fontsize, pad=20)  # Added line break and padding
        plt.xticks(x, all_labels, rotation=0, ha='right', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(fontsize=fontsize, loc='upper right')  # Moved legend inside plot

        plt.ylim(top=max_count*margin_factor)  # Increased top margin

        # Adjust layout with more bottom margin
        plt.subplots_adjust(bottom=0.2)
        
        # Save the plot as PDF
        filename = f"{folder_name}/label_distribution_scenario_{scenario_num_str}_sampling_{splitting_method}.pdf"
        print(f"Saving plot to {filename}")
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close()
