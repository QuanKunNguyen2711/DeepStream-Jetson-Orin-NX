import pandas as pd

def calculate_accuracy(ground_truth_csv, measured_csv):
    gt = pd.read_csv(ground_truth_csv)
    ms = pd.read_csv(measured_csv)
    
    # Traffic Count Accuracy
    traffic_accuracy = {}
    gt_counts = gt.groupby(['lane', 'direction']).size().reset_index(name='Y')
    ms_counts = ms.groupby(['lane', 'direction']).size().reset_index(name='X')
    merged = pd.merge(gt_counts, ms_counts, on=['lane', 'direction'], how='left').fillna(0)
    merged['APE'] = (abs(merged['Y'] - merged['X']) / merged['Y']) * 100
    merged['Accuracy'] = 100 - merged['APE']
    traffic_accuracy['by_direction'] = merged.to_dict('records')
    
    # Vehicle Classification Accuracy
    classification_accuracy = {}
    gt_type = gt.groupby(['lane', 'vehicle_type']).size().reset_index(name='Y')
    ms_type = ms.groupby(['lane', 'vehicle_type']).size().reset_index(name='X')
    merged_type = pd.merge(gt_type, ms_type, on=['lane', 'vehicle_type'], how='left').fillna(0)
    merged_type['Correct'] = merged_type[['Y', 'X']].min(axis=1)
    total_Y = merged_type.groupby('lane')['Y'].sum().reset_index(name='Total_Y')
    merged_total = pd.merge(merged_type, total_Y, on='lane')
    merged_total['PE'] = ((merged_total['Y'] - merged_total['Correct']) / merged_total['Total_Y']) * 100
    merged_total['Accuracy'] = 100 - merged_total['PE']
    classification_accuracy['by_type'] = merged_total.to_dict('records')
    
    return {
        'traffic_accuracy': traffic_accuracy,
        'classification_accuracy': classification_accuracy
    }

results = calculate_accuracy('ground_truth_TranHungDao_NguyenVanCu.csv', 'TranHungDao_NguyenVanCu.csv')
print(results)