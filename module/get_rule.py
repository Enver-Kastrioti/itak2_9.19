import os
import json
import argparse
import re


def _iter_clean_lines(block: str):
    """Yield non-empty, non-comment lines from a block."""
    for line in block.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        yield line


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self):
        t = self.peek()
        if t:
            self.pos += 1
        return t

    def parse(self):
        if not self.tokens:
            return None
        try:
            return self.parse_expression()
        except Exception:
            return None

    def parse_expression(self):
        # Expression = Term { : Term }
        left = self.parse_term()
        while self.peek() == ':':
            self.consume() # eat ':'
            right = self.parse_term()
            if right is None: break
            left_node = {'op': 'OR', 'children': [left, right]}
            if isinstance(left, dict) and left.get('op') == 'OR':
                left_node['children'] = left['children'] + [right]
            left = left_node
        return left

    def parse_term(self):
        # Term = Factor { # Factor }
        left = self.parse_factor()
        while self.peek() == '#':
            self.consume() # eat '#'
            right = self.parse_factor()
            if right is None: break
            left_node = {'op': 'AND', 'children': [left, right]}
            if isinstance(left, dict) and left.get('op') == 'AND':
                left_node['children'] = left['children'] + [right]
            left = left_node
        return left

    def parse_factor(self):
        token = self.peek()
        if token == '(':
            self.consume()
            expr = self.parse_expression()
            if self.peek() == ')':
                self.consume()
            return expr
        elif token and token not in '):#':
            return self.consume()
        return None


def parse_required_logic(text):
    if not text or text == 'NA':
        return None
    # Tokenize: split by ( ) : # but keep them
    regex = r'([():#])'
    parts = re.split(regex, text)
    tokens = [p.strip() for p in parts if p.strip()]
    parser = Parser(tokens)
    return parser.parse()


def parse_rule_file(file_path):
    rules_dict = {}
    # 检查输入文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The input file '{file_path}' was not found.")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        # 将输入文件所有行合并为一个字符串，去除空白字符，按照//分割为多个块
        blocks = ''.join(file.readlines()).strip().split('//')
        
        for block in blocks:
            # 遍历每个块，跳过空的块
            if not block.strip():
                continue
            # 将块分割成多行，去除掉以#开头的注释行
            lines = list(_iter_clean_lines(block))
            if not lines:
                continue
            # 创建一个空字典
            rule = {}
            for line in lines:
                # 使用第一个出现的':'将行分割成键和值两部分
                if ':' not in line:
                    continue
                key, value = line.split(':', 1)
                rule[key.strip()] = value.strip()
            
            # 从字典中获取必要字段
            id_value = rule.get('ID', None)
            if not id_value:
                # 非规则块或缺少ID，跳过
                continue
            
            # 构建规则数据结构
            required_raw = rule.get('Required', '')
            
            # 使用新的逻辑解析器
            logic_tree = parse_required_logic(required_raw)
            
            # 为了兼容性，如果解析失败或结果简单，回退到旧模式?
            # 其实我们可以统一使用 logic 模式，或者只在复杂时使用
            # 但为了统一处理，建议全部视为 logic 模式
            
            # 提取所有涉及的domain用于快速过滤（可选）
            # flat_domains = [r for r in required_raw.replace(':', '#').replace('(', '').replace(')', '').split('#') if r]
            
            rule_data = {
                'id': id_value, # 添加ID字段
                'name': rule.get('Name', 'NA'),
                'family': rule.get('Family', 'NA'),
                'type': rule.get('Type', 'NA'),
                'desc': [] if rule.get('Desc', 'NA') == 'NA' else [rule.get('Desc')],
                'mode': ['logic'], # 统一使用 logic 模式
                'logic': logic_tree,
                'required': [], # 不再使用扁平列表
                'forbidden': rule.get('Forbidden', '').split(':') if 'Forbidden' in rule else []
            }
            
            rules_dict[id_value] = rule_data
    
    return rules_dict


def parse_score_thresholds(file_path):
    """
    解析规则文件中的全局特定条目分数阈值。
    示例行：
      Score:PS50863(10):cd10017(15)
    注意：该阈值是全局性的，与具体家族所属无关；若同一个条目在多个家族下重复出现，则取其中的最大阈值（更严格）。
    返回：dict，例如 {"PS50863": 10.0, "cd10017": 15.0}
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The input file '{file_path}' was not found.")
    
    thresholds = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        blocks = ''.join(file.readlines()).strip().split('//')
        for block in blocks:
            if not block.strip() or block.strip().startswith('#'):
                continue
            # 逐行检查 Score 行
            for line in _iter_clean_lines(block):
                if not line.startswith('Score:'):
                    continue
                # 去掉前缀并按 ':' 分割每个条目
                content = line[len('Score:'):].strip()
                parts = [p for p in content.split(':') if p]
                for part in parts:
                    part = part.strip()
                    # 形如 PS50863(10) 或 cd10017(15)
                    if '(' in part and part.endswith(')'):
                        try:
                            name, val_str = part.split('(', 1)
                            name = name.strip()
                            val = float(val_str[:-1])  # 去除末尾 ')'
                            if name:
                                # 如果重复出现，采用最大阈值以更严格
                                thresholds[name] = max(val, thresholds.get(name, float('-inf')))
                        except Exception:
                            # 忽略无法解析的项
                            continue
    return thresholds


def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='解析规则文件并生成JSON格式输出')
    parser.add_argument('-i', '--input', required=True, help='输入规则文件路径')
    parser.add_argument('-o', '--output', required=True, help='输出目录路径')
    parser.add_argument('--with-score', action='store_true', help='同时输出全局特定条目分数阈值')
    
    args = parser.parse_args()
    
    input_file = args.input
    output_dir = args.output
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 输出文件路径
    output_file = os.path.join(output_dir, "getrule.json")
    
    try:
        # 解析规则文件
        rules = parse_rule_file(input_file)
        
        # 将结果写入JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(rules, f, indent=2, ensure_ascii=False)
        
        # 可选：输出特定条目分数阈值（单独文件，避免破坏下游消费者）
        if args.with_score:
            score_file = os.path.join(output_dir, "score_thresholds.json")
            score_thresholds = parse_score_thresholds(input_file)
            with open(score_file, 'w', encoding='utf-8') as f:
                json.dump(score_thresholds, f, indent=2, ensure_ascii=False)
        
        print(f"规则已成功解析并保存到 {output_file}")
        if args.with_score:
            print(f"分数阈值已保存到 {score_file}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")


if __name__ == "__main__":
    main()
