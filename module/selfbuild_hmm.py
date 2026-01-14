import json
import os
import argparse
from collections import defaultdict

def parse_pfam_spec(file_path):
    # 使用defaultdict来存储结果
    result = defaultdict(lambda: defaultdict(list))
    hit_counts = defaultdict(lambda: defaultdict(int))
    temp_scores = defaultdict(dict)  # 用于临时存储NF-YB和NF-YC的得分
    
    # 读取文件并处理每一行
    with open(file_path, 'r') as f:
        for line in f:
            # 跳过注释行和空行
            if line.startswith('#') or not line.strip():
                continue
            
            # 分割行内容
            fields = line.strip().split()
            if len(fields) < 9:
                continue
                
            # 提取必要的字段
            accession = fields[1]  # 第2列
            gene_id = fields[2]    # 第3列
            evalue = fields[7]     # 第8列
            score = fields[8]      # 第9列
            
            # 检查score值
            try:
                score_value = float(score)
                if score_value < 20:
                    continue
            except ValueError:
                continue

            # 记录命中次数
            hit_counts[gene_id][accession] += 1
            
            # 构建命中记录，格式与IPR数据一致
            hit = {
                "accession": accession,
                "library": "selfbuild",
                "ipr": "",  # hmmscan数据中没有IPR信息，留空
                "ipr_name": "",  # hmmscan数据中没有IPR名称，留空
                "description": accession,  # 使用accession作为描述
                "start": "",  # hmmscan数据中没有起始位置，留空
                "end": "",    # hmmscan数据中没有结束位置，留空
                "evalue": evalue,
                "score": score
            }
            
            # 如果是多重命中，使用&符号标记
            count = hit_counts[gene_id][accession]
            if count > 1:
                # 添加带计数的条目
                key = f"{accession}&{count}"
                result[gene_id][key].append(hit)
                
                # 添加空列表
                for i in range(count-1, 0, -1):
                    empty_key = f"{accession}&{i}" if i > 1 else accession
                    if not result[gene_id][empty_key]:
                        result[gene_id][empty_key] = []
            else:
                result[gene_id][accession].append(hit)
    
    # 将结果转换为与cl_json.json一致的格式
    match_list = []
    for gene_id, gene_data in result.items():
        # 创建与cl_json.json格式一致的结构
        # 第一个元素是空字典（代替序列信息）
        # 第二个元素是域信息字典
        gene_entry = {
            gene_id: [
                {},  # 空字典代替序列信息
                dict(gene_data)  # 域信息字典
            ]
        }
        match_list.append(gene_entry)
    
    formatted_result = {
        "result": {
            "match": match_list
        }
    }
    
    return formatted_result

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='处理PFAM特异性分析结果文件')
    parser.add_argument('-i', '--input', required=True, help='输入文件路径（hmmscan结果文件）')
    parser.add_argument('-o', '--output', required=True, help='输出目录路径')
    
    args = parser.parse_args()
    
    input_file = args.input
    output_dir = args.output
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 输出文件路径
    output_file = os.path.join(output_dir, "pfamspec.json")
    
    try:
        # 解析文件并获取结果
        result = parse_pfam_spec(input_file)
        
        # 将结果写入JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
    except Exception as e:
        raise

if __name__ == "__main__":
    main()