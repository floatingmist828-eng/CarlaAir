import os
import csv
import argparse
import re

def extract_car_metrics(log_file_path):
    """
    提取日志文件中以'car'开头的数字指标
    当car行数>2（即≥3行）时忽略第一行
    返回包含每行指标列表的列表（每个指标作为单独元素）
    """
    all_car_lines = []
    
    try:
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if re.match(r'^car\b', line, re.IGNORECASE):
                    parts = line.split()
                    if len(parts) > 1:
                        # 保存指标列表（去掉"car"后的所有部分）
                        all_car_lines.append(parts[1:])
    
    except Exception as e:
        print(f"Error processing {log_file_path}: {str(e)}")
    
    # 条件处理：当行数>2时忽略第一行
    if len(all_car_lines) > 2:
        return sum(all_car_lines[1:],[])
    return sum(all_car_lines,[])

def determine_fusion_type(file_path):
    """
    根据文件路径确定融合类型
    """
    file_path = file_path.lower()
    
    if 'bev' in file_path:
        return 'bev_fusion'
    elif 'late' in file_path:
        return 'late_fusion'
    elif 'early' in file_path:
        return 'early_fusion'
    else:
        return 'instance_fusion'

def process_logs(root_path):
    """
    处理所有子文件夹中的logs文件夹
    返回包含结果的字典列表
    """
    results = []
    
    for entry in os.scandir(root_path):
        if entry.is_dir():
            logs_dir = os.path.join(entry.path, 'logs')
            
            if os.path.exists(logs_dir) and os.path.isdir(logs_dir):
                for log_file in os.listdir(logs_dir):
                    log_file_path = os.path.join(logs_dir, log_file)
                    
                    if os.path.isfile(log_file_path):
                        rel_path = os.path.relpath(log_file_path, root_path)
                        fusion_type = determine_fusion_type(log_file_path)
                        metrics = extract_car_metrics(log_file_path)
                        if metrics == [] or len(metrics) > 25:
                            continue  # 如果没有指标则跳过
                        
                        record = {
                            'fusion_type': fusion_type,
                            'file_path': log_file_path.replace('\\', '/')
                        }

                        # 为每个指标添加单独列
                        for i, metric in enumerate(metrics, start=1):
                            record[f'metric_{i}'] = metric
                        results.append(record)
    return results

def main():
    parser = argparse.ArgumentParser(description='Extract car metrics with separate columns')
    parser.add_argument('root_path', help='Root directory to scan')
    parser.add_argument('output_csv', help='Output CSV file path')
    args = parser.parse_args()

    data = process_logs(args.root_path)
    
    if data:
        # 动态确定最大指标列数
        max_metrics = max(len(item)-2 for item in data)  # 减去fusion_type和file_path两列
        fieldnames = ['fusion_type', 'file_path'] + [f'metric_{i}' for i in range(1, max_metrics+1)]
        
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        print(f"Successfully processed {len(data)} entries with up to {max_metrics} metrics per line")
        print(f"Output saved to {args.output_csv}")
    else:
        print("No valid car metrics found")

if __name__ == "__main__":
    main()