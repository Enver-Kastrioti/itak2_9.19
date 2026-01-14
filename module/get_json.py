import json
import os
import argparse


def load_input_data(input_file):
    """加载输入JSON文件"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
        return None
    except json.JSONDecodeError as e:
        print(f"错误：JSON解析失败 - {e}")
        return None
    except Exception as e:
        print(f"错误：读取文件失败 - {e}")
        return None


def parse_specific_thresholds_from_rule(rule_file_path):
    """从 rule.txt 中解析所有块内的 'Score:' 行为全局特定条目阈值映射。"""
    thresholds = {}
    try:
        with open(rule_file_path, 'r', encoding='utf-8') as f:
            blocks = ''.join(f.readlines()).strip().split('//')
        for block in blocks:
            block = block.strip()
            if not block or block.startswith('#'):
                continue
            for raw_line in block.split('\n'):
                line = raw_line.strip()
                if not line or line.startswith('#'):
                    continue
                if not line.startswith('Score:'):
                    continue
                content = line[len('Score:'):].strip()
                parts = [p for p in content.split(':') if p]
                for part in parts:
                    part = part.strip()
                    if '(' in part and part.endswith(')'):
                        try:
                            name, val_str = part.split('(', 1)
                            name = name.strip()
                            val = float(val_str[:-1])
                            if name:
                                thresholds[name] = max(val, thresholds.get(name, float('-inf')))
                        except Exception:
                            continue
    except Exception:
        # 解析失败时返回空映射，保持流程继续
        return {}
    return thresholds


def load_specific_thresholds():
    """定位并加载 db/rule.txt 的特定条目阈值映射。"""
    # rule.txt 位于项目根目录下的 db 文件夹；get_json.py 在 module 目录
    project_root = os.path.dirname(os.path.dirname(__file__))
    rule_file = os.path.join(project_root, 'db', 'rule.txt')
    if os.path.exists(rule_file):
        return parse_specific_thresholds_from_rule(rule_file)
    return {}


def is_valid_score(score, score_threshold=1.0, accession=None, specific_thresholds=None):
    """
    检查分数是否有效：
    - 若命中特定条目阈值（按 accession），采用 score >= 特定阈值；
    - 否则采用 score > 全局阈值；
    - 'STRONG' 直接通过。
    """
    if score == "STRONG":
        return True
    try:
        s = float(score)
    except (ValueError, TypeError):
        return False

    if specific_thresholds and accession in specific_thresholds:
        return s >= float(specific_thresholds[accession])
    return s > float(score_threshold)


def get_score_value(score):
    """获取分数值"""
    return 100 if score == "STRONG" else float(score or 0)


def process_ipr_groups(gene_data, gene_id, output_dir, debug=False, score_threshold=1.0, specific_thresholds=None):
    """处理IPR分组数据"""
    new_gene_data = {}
    
    # 创建一个列表来存储 overlap < 0 的情况
    overlap_records = []
    
    for ipr, matches in gene_data[1].items():
        # 过滤掉不满足score条件的匹配，优先考虑特定条目阈值
        valid_matches = [
            m for m in matches
            if is_valid_score(m.get("score"), score_threshold, accession=m.get("accession"), specific_thresholds=specific_thresholds)
        ]
        
        if not valid_matches:
            continue
            
        # 按accession和library分组
        groups = {}
        for match in valid_matches:
            key = (match["accession"], match["library"])
            if key not in groups:
                groups[key] = []
            groups[key].append(match)
        
        # 处理每个分组并保存结果
        final_matches = []
        max_group_size = 0
        
        for (acc, lib), group in groups.items():
            # 按start值排序
            group.sort(key=lambda x: x["start"])
            
            # 检查重叠
            i = 0
            while i < len(group):
                j = i + 1
                while j < len(group):
                    # 计算重叠范围
                    overlap = group[j]["start"] - group[i]["end"] - 1
                    
                    # 记录所有 overlap < 0 的情况
                    if overlap < 0:
                        overlap_record = {
                            "gene_id": gene_id,
                            "ipr": ipr,
                            "accession": acc,
                            "library": lib,
                            "match1": {
                                "start": group[i]["start"],
                                "end": group[i]["end"],
                                "score": group[i]["score"]
                            },
                            "match2": {
                                "start": group[j]["start"],
                                "end": group[j]["end"],
                                "score": group[j]["score"]
                            },
                            "overlap": overlap
                        }
                        overlap_records.append(overlap_record)
                        # 只在debug模式下写入overlop.txt到输出目录
                        if debug:
                            with open(os.path.join(output_dir, "overlop.txt"), "a") as f:
                                f.write(f"Gene ID: {gene_id}\n")
                                f.write(f"IPR: {ipr}\n")
                                f.write(f"Accession: {acc}\n")
                                f.write(f"Library: {lib}\n")
                                f.write(f"Match 1: start={group[i]['start']}, end={group[i]['end']}, score={group[i]['score']}\n")
                                f.write(f"Match 2: start={group[j]['start']}, end={group[j]['end']}, score={group[j]['score']}\n")
                                f.write(f"Overlap: {overlap}\n")
                                f.write("---\n")
                    
                    if overlap < -100:
                        # 比较score，移除score较小的
                        score_i = get_score_value(group[i]["score"])
                        score_j = get_score_value(group[j]["score"])
                        
                        if score_i < score_j:
                            group.pop(i)
                            j = i + 1  # 重置j
                            continue
                        else:
                            group.pop(j)
                            continue
                    j += 1
                i += 1
            
            # 更新最大分组大小
            current_group_size = len(group)
            max_group_size = max(max_group_size, current_group_size)
            
            # 添加处理后的匹配
            final_matches.extend([
                {
                    "accession": m["accession"],
                    "library": m["library"],
                    "start": m["start"],
                    "end": m["end"],
                    "evalue": m["evalue"],
                    "score": m["score"]
                } for m in group
            ])
        
        # 添加主匹配组和递减的空表格
        if final_matches:
            if max_group_size > 2:
                # 添加主匹配组（使用最大数值）
                new_gene_data[f"{ipr}&{max_group_size}"] = final_matches
                # 添加递减的空表格
                for i in range(max_group_size-1, 1, -1):
                    new_gene_data[f"{ipr}&{i}"] = []
                # 添加最后一个不带数字的空表格
                new_gene_data[ipr] = []
            elif max_group_size == 2:
                # 如果最大分组大小为2
                new_gene_data[f"{ipr}&2"] = final_matches
                new_gene_data[ipr] = []
            else:
                # 如果最大分组大小为1，直接使用原始IPR名称
                new_gene_data[ipr] = final_matches
    
    # 只在debug模式下将 overlap < 0 的记录写入文件到输出目录
    if overlap_records and debug:
        with open(os.path.join(output_dir, "overlop.txt"), "a") as f:
            for record in overlap_records:
                f.write(f"Gene ID: {record['gene_id']}\n")
                f.write(f"IPR: {record['ipr']}\n")
                f.write(f"Accession: {record['accession']}, Library: {record['library']}\n")
                f.write(f"Match 1: Start={record['match1']['start']}, End={record['match1']['end']}, Score={record['match1']['score']}\n")
                f.write(f"Match 2: Start={record['match2']['start']}, End={record['match2']['end']}, Score={record['match2']['score']}\n")
                f.write(f"Overlap: {record['overlap']}\n")
                f.write("---\n")
    
    return new_gene_data
#从json中读取数据

def process_data(input_file, output_dir=None, debug=False, score_threshold=1.0):
    # 加载输入数据
    old_data = load_input_data(input_file)
    if old_data is None:
        return None, None
    
    # 加载全局特定条目分数阈值映射（来自 db/rule.txt 的 Score 行）
    specific_thresholds = load_specific_thresholds()

    # 构建新结构
    new_result = {
        "result": {
            "match": []
        }
    }

    # 新格式的结果
    filtered_result = {
        "result": {
            "match": []
        }
    }

    # 如果提供了输出目录，确保目录存在
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # 只在debug模式下清空overlop.txt文件
        if debug:
            with open(os.path.join(output_dir, "overlop.txt"), "w") as f:
                f.write("")

    for item in old_data.get("results", []):
        gene_data_list = []
        filtered_gene_data_list = []
        
        for xref in item.get("xref", []):
            gene_id = xref["id"]
            
            if item.get("matches"):
                # 处理原始格式
                gene_data = [
                    {"sequence": item["sequence"]},
                    {}
                ]

                for match in item.get("matches", []):
                    signature = match.get("signature", {})
                    signature_accession = signature.get("accession")
                    library = signature.get("signatureLibraryRelease", {}).get("library")
                    
                    entry = signature.get("entry")
                    if not isinstance(entry, dict):
                        continue
                    
                    ipr_accession = entry.get("accession")
                    if not ipr_accession:
                        continue

                    ipr_name = entry.get("name")
                    description = entry.get("description")
                    parent_score = match.get("score")
                    
                    for location in match["locations"]:
                        start = location.get("start")
                        end = location.get("end")
                        score = location.get("score", location.get("level", parent_score))
                        
                        if "evalue" in location:
                            evalue = location["evalue"]
                        else:
                            evalue = location.get("representative", None)
                        
                        match_item = {
                            "accession": signature_accession,
                            "library": library,
                            "ipr": ipr_accession or "null",
                            "ipr_name": ipr_name or "null",
                            "description": description or "null",
                            "start": start or "null",
                            "end": end or "null",
                            "evalue": evalue or "null",
                            "score": score or "null"
                        }

                        if ipr_accession not in gene_data[1]:
                            gene_data[1][ipr_accession] = []
                        gene_data[1][ipr_accession].append(match_item)
                
                # 添加到原始格式列表
                gene_data_list.append({gene_id: gene_data})
                
                # 处理新格式（传递特定条目阈值映射）
                filtered_data = process_ipr_groups(gene_data, gene_id, output_dir or "", debug, score_threshold, specific_thresholds=specific_thresholds)
                if filtered_data:
                    # 构建与test版本一致的数据结构
                    filtered_gene_data = [
                        {"sequence": gene_data[0]["sequence"]},
                        filtered_data
                    ]
                    filtered_gene_data_list.append({gene_id: filtered_gene_data})
            else:
                pass  # 没有matches数据，跳过处理

        # 合并所有基因数据
        if gene_data_list:
            new_result["result"]["match"].extend(gene_data_list)
        if filtered_gene_data_list:
            filtered_result["result"]["match"].extend(filtered_gene_data_list)

    # 如果提供了输出目录，保存文件
    if output_dir:
        with open(os.path.join(output_dir, "processed_ipr_domains.json"), "w") as f:
            json.dump(filtered_result, f, indent=2)

        # 只在debug模式下保存调试文件
        if debug:
            with open(os.path.join(output_dir, "raw_interproscan_data.json"), "w") as f:
                json.dump(new_result, f, indent=2)
            print("转换完成，结果已保存为 processed_ipr_domains.json 和 raw_interproscan_data.json（调试模式）")
        else:
            print("转换完成，结果已保存为 processed_ipr_domains.json")
    else:
        print("数据处理完成，返回内存中的结果")
    
    # 返回处理后的数据
    return filtered_result, new_result


def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='处理InterProScan JSON文件并生成新格式输出')
    parser.add_argument('-i', '--input', required=True, help='输入JSON文件路径（InterProScan输出）')
    parser.add_argument('-o', '--output', required=True, help='输出目录路径')
    parser.add_argument('--debug', action='store_true', help='启用调试模式，输出调试文件')
    parser.add_argument('--score', type=float, default=1.0, help='score阈值，只有大于此值的结果才会被保留 (默认: 1.0)')
    
    args = parser.parse_args()
    
    # 调用process_data函数
    filtered_result, new_result = process_data(
        input_file=args.input,
        output_dir=args.output,
        debug=args.debug,
        score_threshold=args.score
    )
    
    if filtered_result is None:
        print("数据处理失败")
        return False
    
    return True


if __name__ == "__main__":
    main()