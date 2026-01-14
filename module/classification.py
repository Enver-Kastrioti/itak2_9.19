import json
import os
import sys
import importlib.util
from pathlib import Path

# 动态导入getrule和specpfam模块
def import_module_dynamically(module_name):
    """动态导入模块"""
    try:
        # 首先尝试直接导入
        return importlib.import_module(module_name)
    except ImportError:
        # 如果失败，尝试从当前目录导入
        current_dir = Path(__file__).parent
        module_path = current_dir / f"{module_name}.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        else:
            raise ImportError(f"Cannot find module {module_name}")

# 导入所需模块
get_rule = import_module_dynamically('get_rule')
selfbuild_hmm = import_module_dynamically('selfbuild_hmm')

# 获取函数引用
parse_rule_file = get_rule.parse_rule_file
parse_pfam_spec = selfbuild_hmm.parse_pfam_spec

# 全局变量，将在运行时设置
filtered_result = None
result_dir = None

def set_result_dir(dir_path):
    """设置结果目录路径"""
    global result_dir
    result_dir = dir_path

def load_filtered_result(filtered_data=None):
    """加载过滤后的结果，优先使用内存中的数据，否则从文件读取
    
    Args:
        filtered_data (dict, optional): 内存中的过滤数据
    
    Returns:
        dict: 加载的数据，如果失败则返回默认结构
    """
    global result_dir
    
    # 如果提供了内存中的数据，直接使用
    if filtered_data is not None:
        print("使用内存中的过滤数据")
        return filtered_data
    
    # 否则从文件读取（保持向后兼容）
    if not result_dir:
        print("[错误] 结果目录未设置")
        return {"result": {"match": []}}
        
    processed_ipr_path = os.path.join(result_dir, "processed_ipr_domains.json")
    
    if os.path.exists(processed_ipr_path):
        print(f"从文件加载过滤数据: {processed_ipr_path}")
        with open(processed_ipr_path, "r") as f:
            return json.load(f)
    else:
        print(f"[错误] 找不到processed_ipr_domains.json文件: {processed_ipr_path}")
        return {"result": {"match": []}}

def initialize_module(result_directory, filtered_data=None):
    """初始化模块，设置结果目录并加载数据
    
    Args:
        result_directory (str): 结果目录路径
        filtered_data (dict, optional): 内存中的过滤数据
    """
    global filtered_result
    set_result_dir(result_directory)
    filtered_result = load_filtered_result(filtered_data)

# 获取当前文件所在目录的路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 全局变量，将在运行时设置
rules_dict = {}
result_spec = {}

def initialize_rules(rule_file_path):
    """初始化规则文件"""
    global rules_dict
    try:
        if not os.path.exists(rule_file_path):
            raise FileNotFoundError(f"规则文件不存在：{rule_file_path}")
            
        rules_dict = parse_rule_file(rule_file_path)
        print("[成功] 成功加载规则文件")
        return True
    except Exception as e:
        print(f"[错误] 错误：加载规则文件时发生异常 - {str(e)}")
        rules_dict = {}
        return False

def merge_results(dict1, dict2):
    """合并两个结果字典"""
    print(f"[信息] 开始合并结果...")
    
    # 创建新的match列表
    merged_matches = []
    
    # 将第一个字典的所有基因添加到merged_matches
    for gene_match in dict1["result"]["match"]:
        merged_matches.append(gene_match)
    
    # 将第二个字典的基因添加或合并到merged_matches
    merged_count = 0
    new_count = 0
    
    for gene_match in dict2["result"]["match"]:
        for gene_id, gene_data in gene_match.items():
            # 检查这个基因是否已经存在
            found = False
            for existing_match in merged_matches:
                if gene_id in existing_match:
                    # 找到了现有基因，合并域数据
                    existing_gene_data = existing_match[gene_id]
                    
                    # 两种格式现在都是 [sequence_obj, {domain: [matches]}]
                    if isinstance(existing_gene_data, list) and len(existing_gene_data) >= 2:
                        if isinstance(existing_gene_data[1], dict) and isinstance(gene_data, list) and len(gene_data) >= 2:
                            # 合并第二个元素（域字典）
                            for domain, hits in gene_data[1].items():
                                if domain in existing_gene_data[1]:
                                    # 如果域已存在，扩展匹配列表
                                    if isinstance(existing_gene_data[1][domain], list):
                                        existing_gene_data[1][domain].extend(hits)
                                    else:
                                        existing_gene_data[1][domain] = [existing_gene_data[1][domain]] + hits
                                else:
                                    # 新域，直接添加
                                    existing_gene_data[1][domain] = hits
                    
                    merged_count += 1
                    found = True
                    break
            
            if not found:
                # 如果是新基因，创建新的字典并添加
                merged_matches.append({gene_id: gene_data})
                new_count += 1
    
    print(f"[成功] 合并完成: {merged_count} 个基因合并, {new_count} 个新基因添加")
    return {"result": {"match": merged_matches}}

# 其他函数保持不变
def get_ipr_counts(gene_data):
    """从基因数据中提取IPR计数
    注意：同一个IPR下可能包含来自不同数据库（accession）的hits。
    计算hits数量时，应取单个accession出现的最大次数，而不是所有hits的总和。
    """
    counts = {}
    if len(gene_data) > 1 and isinstance(gene_data[1], dict):
        for ipr_key, hits in gene_data[1].items():
            # 处理键名中可能存在的计数后缀（如 IPR001005&3）
            real_ipr = ipr_key.split('&')[0]
            
            # 统计每个accession出现的次数
            accession_counts = {}
            if isinstance(hits, list):
                for hit in hits:
                    acc = hit.get('accession')
                    if acc:
                        accession_counts[acc] = accession_counts.get(acc, 0) + 1
            
            # 取最大值作为该IPR的有效拷贝数
            # 如果没有hits，则为0
            num_hits = max(accession_counts.values()) if accession_counts else 0
            
            # 累加计数（防止同一个IPR被拆分到多个键中，虽然理论上不应发生，但为了稳健）
            # 注意：如果拆分到多个键，取最大值可能不准确，这里假设同一IPR的所有hits都在一个键下
            # 或者如果已经拆分了，我们应该合并hits列表后再计算。
            # 鉴于输入数据的结构，同一个IPR（不带后缀）通常在一个键下。
            # 如果是带后缀的键，说明上游已经做过某种预处理，但为了安全，我们还是累加到real_ipr下，
            # 但这里有个问题：如果是分批进来的，简单的累加可能会导致跨accession的错误合并。
            # 更稳妥的做法是：维护一个全局的 {real_ipr: {accession: count}} 结构。
            
            if real_ipr not in counts:
                counts[real_ipr] = {}
            
            for acc, count in accession_counts.items():
                counts[real_ipr][acc] = counts[real_ipr].get(acc, 0) + count
                
    # 将 {ipr: {acc: count}} 转换为 {ipr: max_count}
    final_counts = {}
    for ipr, acc_dict in counts.items():
        final_counts[ipr] = max(acc_dict.values()) if acc_dict else 0
        
    return final_counts

def evaluate_logic(node, ipr_counts):
    """递归评估逻辑树"""
    if node is None:
        return True # 无要求则视为匹配
        
    if isinstance(node, str):
        # 叶节点：检查具体domain
        domain = node
        required_count = 1
        # 处理计数后缀，例如 IPR001471&2
        if '&' in domain:
            parts = domain.split('&')
            domain = parts[0]
            if len(parts) > 1 and parts[1].isdigit():
                required_count = int(parts[1])
        
        actual_count = ipr_counts.get(domain, 0)
        return actual_count >= required_count

    if isinstance(node, dict):
        op = node.get('op')
        children = node.get('children', [])
        
        if op == 'AND':
            return all(evaluate_logic(child, ipr_counts) for child in children)
        elif op == 'OR':
            return any(evaluate_logic(child, ipr_counts) for child in children)
            
    return False

def check_rule_match(ipr_counts, rule_data):
    """检查IPR计数是否匹配规则"""
    # 检查forbidden条件
    forbidden = rule_data.get("forbidden", [])
    for f in forbidden:
        if f == 'NA': continue
        if f in ipr_counts:
            return False
    
    # 检查逻辑规则
    logic_tree = rule_data.get("logic")
    if logic_tree is not None:
        return evaluate_logic(logic_tree, ipr_counts)
        
    # 如果没有逻辑树（兼容旧模式，虽然现在get_rule都生成逻辑树）
    mode = rule_data.get("mode", [])[0] if rule_data.get("mode") else None
    
    # 如果也没有mode或mode是logic但tree是None，说明无required条件，视为匹配
    if not mode or mode == "logic":
        return True

    # 旧模式回退（理论上不会执行到这里）
    required = set(rule_data.get("required", []))
    ipr_set = set(ipr_counts.keys())
    
    if mode == "a":
        return required.issubset(ipr_set)
    elif mode == "b":
        return not required.isdisjoint(ipr_set)
    
    return False

def classify_genes(input_dict, mode='specific'):
    """对基因进行分类并生成结果
    
    Args:
        input_dict (dict): 输入的基因匹配数据
        mode (str): 分类模式，'specific' (特异性优先) 或 'score' (得分优先)
    """
    result = {}
    
    print(f"[信息] 使用分类模式: {mode}")
    
    # 处理输入字典中的每个基因
    for gene_match in input_dict["result"]["match"]:
        for gene_id, gene_data in gene_match.items():
            # 获取基因的IPR计数
            ipr_counts = get_ipr_counts(gene_data)
            
            # 存储匹配的所有家族
            matched_families = []
            final_rule = None
            
            # 检查每个规则
            for rule_id, rule_data in rules_dict.items():
                if check_rule_match(ipr_counts, rule_data):
                    matched_families.append(rule_data)
            
            # 如果有匹配的规则
            if matched_families:
                # 选择优先级最高的规则作为主要规则
                # 规则文件中越靠后的规则通常越具体，但也存在例外
                # 在这里，我们需要根据规则的特异性来选择
                # 例如，Required条件越多的规则，优先级越高
                # 或者，Required条件包含数量限制的规则（如&2），优先级高于普通规则
                
                # 获取每个IPR的最高score
                ipr_max_scores = {}
                # 预处理：计算每个IPR的最大Score
                if len(gene_data) > 1 and isinstance(gene_data[1], dict):
                    for ipr_key, hits in gene_data[1].items():
                        real_ipr = ipr_key.split('&')[0]
                        max_score = 0.0
                        if isinstance(hits, list):
                            for hit in hits:
                                score_val = hit.get('score', 0)
                                if str(score_val).upper() == 'STRONG':
                                    score_val = 100.0
                                else:
                                    try:
                                        score_val = float(score_val)
                                    except:
                                        score_val = 0.0
                                if score_val > max_score:
                                    max_score = score_val
                        # 如果同一个real_ipr在多个键中出现，取最大的
                        if max_score > ipr_max_scores.get(real_ipr, 0):
                            ipr_max_scores[real_ipr] = max_score

                def calculate_rule_score_new(rule):
                    # 1. 规则分类：a类 vs b类
                    # a类：只包含单一结构域或只包含OR连接的单一结构域
                    # b类：包含AND连接，或包含数量限制（&N, N>1）
                    
                    if rule.get('name') == 'Others':
                        # Others 特殊处理：权重0.1，作为最后的保底
                        # score设为0.1，确保比任何正常的a类或b类都低
                        return 0.1, 0.1 
                    
                    logic = rule.get('logic')
                    
                    is_b_class = False
                    hit_weight = 0
                    total_score = 0.0
                    
                    def analyze_logic(node):
                        nonlocal is_b_class, hit_weight, total_score
                        
                        if isinstance(node, str):
                            # 叶节点
                            domain = node
                            count = 1
                            if '&' in node:
                                try:
                                    count = int(node.split('&')[1])
                                except:
                                    pass
                            
                            real_ipr = domain.split('&')[0]
                            
                            if count > 1:
                                is_b_class = True
                                hit_weight += count
                                # 计算分数：取前count大的score之和（简化起见，这里目前只取了最大的score，如果需要取前N大，需要修改上游ipr_max_scores的逻辑）
                                # 根据用户描述：IPR001781&2，需使用最大的score和第二大的score的加和
                                # 这需要我们回到原始数据去获取所有score列表
                                
                                # 重新获取该domain的所有score并排序
                                scores = []
                                if len(gene_data) > 1 and isinstance(gene_data[1], dict):
                                    for k, hits in gene_data[1].items():
                                        if k.split('&')[0] == real_ipr:
                                            if isinstance(hits, list):
                                                for hit in hits:
                                                    s = hit.get('score', 0)
                                                    if str(s).upper() == 'STRONG': s = 100.0
                                                    try: s = float(s)
                                                    except: s = 0.0
                                                    scores.append(s)
                                scores.sort(reverse=True)
                                # 取前count个
                                total_score += sum(scores[:count])
                                
                            else:
                                # 单一结构域，a类特征（除非被AND包裹）
                                hit_weight += 1
                                total_score += ipr_max_scores.get(real_ipr, 0)
                                
                        elif isinstance(node, dict):
                            op = node.get('op')
                            children = node.get('children', [])
                            
                            if op == 'AND':
                                is_b_class = True
                                for child in children:
                                    analyze_logic(child)
                            elif op == 'OR':
                                # OR连接，取子规则中得分最高的那个（模拟用户描述：IPR000315:IPR049808，使用得分最高的那个）
                                # 这里稍微复杂，因为我们要判断匹配了哪个分支
                                # 简化处理：遍历所有子节点，看哪个匹配了且分数最高
                                max_child_score = -1.0
                                best_child_weight = 0
                                child_is_b = False
                                
                                # 这里的逻辑稍微有点递归的复杂性，因为OR里面可能嵌套AND
                                # 但根据定义，a类规则是 "只包含OR连接的单一结构域"
                                # 所以如果OR里面有AND，那整个OR作为整体在上一层AND里可能就是b类的一部分
                                # 我们先分别计算每个child的score和weight
                                
                                # 暂存当前状态
                                old_score = total_score
                                old_weight = hit_weight
                                old_b = is_b_class
                                
                                best_branch_score = 0
                                best_branch_weight = 0
                                best_branch_b = False
                                branch_matched = False
                                
                                for child in children:
                                    # 重置累加器用于计算该分支
                                    total_score = 0
                                    hit_weight = 0
                                    is_b_class = False
                                    
                                    # 检查该分支是否匹配（需要使用 evaluate_logic）
                                    if evaluate_logic(child, ipr_counts):
                                        analyze_logic(child)
                                        branch_matched = True
                                        if total_score > best_branch_score:
                                            best_branch_score = total_score
                                            best_branch_weight = hit_weight
                                            best_branch_b = is_b_class
                                
                                # 恢复累加器并加上最佳分支的结果
                                total_score = old_score + best_branch_score
                                hit_weight = old_weight + best_branch_weight
                                if best_branch_b: is_b_class = True # 如果最佳分支是b类，则整体变b类（虽在OR里很少见）
                                is_b_class = is_b_class or old_b

                    if logic:
                        analyze_logic(logic)
                    
                    # 返回 (权重, 总分)
                    # 如果是a类，权重设为1
                    final_weight = hit_weight if is_b_class else 1
                    return final_weight, total_score

                # 计算所有匹配规则的 (权重, 总分)
                rule_metrics = []
                for rule in matched_families:
                    weight, score = calculate_rule_score_new(rule)
                    
                    # 在score模式下，强制所有非Others规则的权重为1，使其只比较得分
                    if mode == 'score' and rule.get('name') != 'Others':
                        weight = 1.0
                        
                    rule_metrics.append({
                        'rule': rule,
                        'id': rule.get('id'), # 使用ID作为唯一标识
                        'weight': weight,
                        'score': score,
                        'is_others': rule.get('name') == 'Others'
                    })
                
                # 过滤掉 a 类规则（权重=1），如果存在 b 类规则（权重>1）
                # 注意：Others 权重为 0.1，不会被视作 b 类，也会被正常的 a 类 (权重1) 压制
                
                max_weight = max(m['weight'] for m in rule_metrics)
                
                # 筛选出权重等于最大权重的规则
                candidates = [m for m in rule_metrics if m['weight'] == max_weight]
                
                # 如果有多个候选，比较 score
                candidates.sort(key=lambda x: x['score'], reverse=True)
                
                # 取 score 最高的
                best_candidate = candidates[0]
                final_rule = best_candidate['rule']
                
                # 收集所有匹配家族的名称
                # 不再输出所有匹配家族，仅保留最终分类
                # all_families = [rule["family"] for rule in matched_families]
                # other_families = ", ".join(all_families)
                
                # 构建结果字典
                # 使用 Name 而不是 Family，但在处理过程中使用 ID 区分
                result[gene_id] = {
                    "name": final_rule["name"],
                    "family": final_rule["name"], # Family字段也使用Name填充，保持一致
                    "type": final_rule["type"],
                    "desc": final_rule.get("desc", []),
                    "other_family": "NA" # 不再显示其他家族
                }
    
    # 将结果写入文件（仅在直接调用时，不在process_with_data模式下）
    try:
        # 只有在result_dir未设置时才使用默认路径（向后兼容）
        if result_dir is None:
            # 构建JSON输出路径
            json_path = os.path.join(os.path.dirname(CURRENT_DIR), 'match.json')
            # 构建TBL输出路径
            tbl_path = os.path.join(os.path.dirname(CURRENT_DIR), 'match_tbl.txt')
            
            # 写入JSON文件
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
            # 写入TBL文件
            with open(tbl_path, 'w', encoding='utf-8') as f:
                for gene_id, data in result.items():
                    desc_str = ';'.join(data['desc']) if data['desc'] else 'NA'
                    line = f"{gene_id}\t{data['name']}\t{data['family']}\t{data['type']}\t{desc_str}\t{data['other_family']}\n"
                    f.write(line)
                
        print("[成功] 结果已成功保存到文件")
    except Exception as e:
        print(f"[错误] 错误：保存结果时发生异常 - {str(e)}")
        
    return result

def process_with_data(result_directory, rule_file, filtered_data=None, spec_data=None, debug=False, mode='specific'):
    """
    使用内存中的数据进行转录因子分类处理
    
    Args:
        result_directory (str): 结果目录路径
        rule_file (str): 规则文件路径
        filtered_data (dict, optional): 内存中的过滤数据
        spec_data (dict, optional): 内存中的spec数据
        debug (bool): 是否启用调试模式
        mode (str): 分类模式，'specific' (默认) 或 'score'
    
    Returns:
        dict: 分类结果
    """
    # 初始化模块
    initialize_module(result_directory, filtered_data)
    
    # 初始化规则
    if not initialize_rules(rule_file):
        print("初始化规则文件失败")
        return None
    
    # 获取filtered_result
    if filtered_data is not None:
        current_filtered_result = filtered_data
    else:
        current_filtered_result = load_filtered_result()
    
    if current_filtered_result is None:
        print("无法获取过滤数据")
        return None
    
    # 处理spec数据
    if spec_data is not None:
        print("使用内存中的spec数据")
        current_spec_result = spec_data
    elif debug:
        # 在debug模式下尝试从文件加载spec数据
        pfamspec_file = os.path.join(result_directory, 'pfamspec.json')
        if os.path.exists(pfamspec_file):
            print(f"从文件加载spec数据: {pfamspec_file}")
            with open(pfamspec_file, 'r') as f:
                current_spec_result = json.load(f)
        else:
            print("debug模式下未找到pfamspec.json文件，使用空数据")
            current_spec_result = {"result": {"match": []}}
    else:
        # 非debug模式下使用空数据
        current_spec_result = {"result": {"match": []}}
    
    # 合并结果
    merged_dict = merge_results(current_filtered_result, current_spec_result)
    
    # 进行分类
    classification_result = classify_genes(merged_dict, mode=mode)
    
    return classification_result

if __name__ == "__main__":
    # 合并filtered_result和result_spec
    merged_dict = merge_results(filtered_result, result_spec)
    
    # 使用合并后的字典进行分类
    classify_genes(merged_dict)
    print("[成功] 分类完成，结果已保存为 match.json")
