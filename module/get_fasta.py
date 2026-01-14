import os
import json
import argparse
from Bio import SeqIO
from Bio.Seq import Seq

def get_output_dir(base_dir=None):
    """获取输出目录路径
    
    Args:
        base_dir (str, optional): 基础目录路径，如果不指定则使用项目根目录
    
    Returns:
        str: result目录的完整路径
    """
    if base_dir is None:
        # 获取当前脚本所在目录的父目录（项目根目录）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(current_dir)
    
    # 创建result目录路径
    output_dir = os.path.join(base_dir, "result")
    
    # 如果目录不存在，创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    return output_dir

def get_output_subdir(name, base_dir=None):
    base = get_output_dir(base_dir)
    sub = os.path.join(base, name)
    if not os.path.exists(sub):
        os.makedirs(sub)
    return sub

def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_project_output_dir(fasta_file):
    root = _repo_root()
    output_base = os.path.join(root, "output")
    base = os.path.splitext(os.path.basename(fasta_file))[0]
    if not os.path.exists(output_base):
        os.makedirs(output_base)
    candidates = []
    for name in os.listdir(output_base):
        if name == base or (name.startswith(base + "_")):
            full = os.path.join(output_base, name)
            if os.path.isdir(full):
                candidates.append(full)
    if candidates:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]
    target = os.path.join(output_base, base)
    os.makedirs(target, exist_ok=True)
    return target

def read_fasta_to_dict(fasta_file):
    """
    读取FASTA文件并转换为字典格式
    
    Args:
        fasta_file (str): FASTA文件路径
    
    Returns:
        dict: 基因ID到序列的映射字典
    """
    fasta_dict = {}
    
    try:
        with open(fasta_file, 'r') as handle:
            for record in SeqIO.parse(handle, "fasta"):
                # 使用record.id作为键，序列作为值
                fasta_dict[record.id] = str(record.seq)
        
        print(f"成功读取 {len(fasta_dict)} 个序列")
        return fasta_dict
        
    except Exception as e:
        print(f"读取FASTA文件时出错: {e}")
        return {}

def format_tf_fasta_with_classification(fasta_file, classification_result, output_file):
    """
    根据分类结果生成包含TF分类信息的标准格式FASTA文件
    
    Args:
        fasta_file (str): 输入的FASTA文件路径
        classification_result (dict): 分类结果字典，格式为 {gene_id: {name, family, type, desc, other_family}}
        output_file (str): 输出FASTA文件路径
    
    Returns:
        bool: 是否成功生成文件
    """
    try:
        fasta_dict = read_fasta_to_dict(fasta_file)
        if not fasta_dict:
            print("无法读取分类输入FASTA文件")
            return False
        
        # 生成格式化的FASTA文件，只包含有分类结果的序列
        with open(output_file, 'w') as f:
            written_count = 0
            for gene_id, sequence in fasta_dict.items():
                # 只处理在分类字典中的基因
                if gene_id in classification_result:
                    tf_info = classification_result[gene_id]
                    
                    # 构建FASTA头部信息
                    # 格式: >gene_id | family | type
                    header = f">{gene_id} | {tf_info['family']} | {tf_info['type']}"
                    
                    # 写入头部和序列
                    f.write(header + '\n')
                    f.write(sequence + '\n')
                    written_count += 1
            
            print(f"成功生成包含 {written_count} 个分类序列的FASTA文件")
            print(f"所有序列都包含TF分类信息")
            return True
            
    except Exception as e:
        print(f"生成FASTA文件时出错: {e}")
        return False

def generate_classified_fasta(fasta_file, classification_result, output_file=None, output_dir=None):
    """
    供其他模块调用的函数，生成包含TF分类信息的FASTA文件
    只包含有分类结果的序列
    
    Args:
        fasta_file (str): 输入的FASTA文件路径
        classification_result (dict): 从内存中获取的分类结果字典
        output_file (str, optional): 输出文件路径，如果不指定则自动生成
        output_dir (str, optional): 输出目录路径，主脚本可以指定项目子文件夹
    
    Returns:
        str: 输出文件路径，如果失败返回None
    """
    try:
        # 如果没有指定输出文件，自动生成路径
        if not output_file:
            input_name = os.path.splitext(os.path.basename(fasta_file))[0]
            if output_dir:
                result_dir = output_dir
            else:
                result_dir = get_output_dir(output_dir)
            output_file = os.path.join(result_dir, f"{input_name}_tf_classified.fasta")
        
        # 调用格式化函数
        success = format_tf_fasta_with_classification(fasta_file, classification_result, output_file)
        
        if success:
            return output_file
        else:
            return None
            
    except Exception as e:
        print(f"生成分类FASTA文件时出错: {e}")
        return None

def _looks_like_cds(seq):
    s = seq.upper()
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    nuc = set("ACGTUNWSMKRYBDHV")
    # 严格判定：全部字符均为核苷酸（移除长度为3的倍数的限制，防止部分CDS被漏掉）
    if not set(letters).issubset(nuc):
        return False
    return True

def _best_six_frame_translate(seq, table=1, min_orf_aa=30):
    s = Seq(seq)
    candidates = []
    for frame in (0, 1, 2):
        aa = str(s[frame:].translate(table=table, to_stop=False))
        seg = max(aa.split('*'), key=len) if aa else ""
        candidates.append((seg, f"+{frame+1}"))
    rc = s.reverse_complement()
    for frame in (0, 1, 2):
        aa = str(rc[frame:].translate(table=table, to_stop=False))
        seg = max(aa.split('*'), key=len) if aa else ""
        candidates.append((seg, f"-{frame+1}"))
    candidates.sort(key=lambda x: len(x[0]), reverse=True)
    best = candidates[0]
    if len(best[0]) < min_orf_aa:
        return "", best[1]
    return best

def _six_frame_translate_all(seq, table=1):
    s = Seq(seq)
    results = []
    for frame in (0, 1, 2):
        aa = str(s[frame:].translate(table=table, to_stop=False)).replace('*', '')
        results.append((aa, f"+{frame+1}"))
    rc = s.reverse_complement()
    for frame in (0, 1, 2):
        aa = str(rc[frame:].translate(table=table, to_stop=False)).replace('*', '')
        results.append((aa, f"-{frame+1}"))
    return results

def generate_protein_sequences_in_memory(fasta_file, genetic_code=1, min_orf_aa=30):
    try:
        sequences = []
        with open(fasta_file, 'r') as handle:
            for record in SeqIO.parse(handle, "fasta"):
                seq = str(record.seq)
                if _looks_like_cds(seq):
                    translated = _six_frame_translate_all(seq, table=genetic_code)
                    for aa, frame in translated:
                        sequences.append({"header": f"{record.id}|frame={frame}", "sequence": aa})
                else:
                    sequences.append({"header": record.id, "sequence": seq})
        return sequences
    except Exception as e:
        print(f"6转翻译生成内存序列失败: {e}")
        return []

def format_tf_fasta_with_classification_from_mem(seqs_dict, classification_result, output_file):
    try:
        with open(output_file, 'w') as f:
            written_count = 0
            for gene_id, sequence in seqs_dict.items():
                if gene_id in classification_result:
                    tf_info = classification_result[gene_id]
                    header = f">{gene_id} | {tf_info['family']} | {tf_info['type']}"
                    f.write(header + '\n')
                    f.write(sequence + '\n')
                    written_count += 1
        print(f"成功生成包含 {written_count} 个分类序列的FASTA文件")
        print(f"所有序列都包含TF分类信息")
        return True
    except Exception as e:
        print(f"生成FASTA文件(内存)时出错: {e}")
        return False
def get_processed_fasta_path(fasta_file, output_dir=None):
    input_name = os.path.splitext(os.path.basename(fasta_file))[0]
    project_dir = get_project_output_dir(fasta_file) if output_dir is None else output_dir
    # processed_dir = os.path.join(project_dir, "six_frame_translation")
    # if not os.path.exists(processed_dir):
    #     os.makedirs(processed_dir)
    # return os.path.join(processed_dir, f"{input_name}_protein_replaced.fasta")
    # 修改为直接在output目录下生成，不再创建six_frame_translation子目录
    return os.path.join(project_dir, f"{input_name}_protein_replaced.fasta")

def generate_protein_fasta_with_translation(fasta_file, output_file=None, output_dir=None, genetic_code=1, min_orf_aa=30):
    try:
        if not output_file:
            output_file = get_processed_fasta_path(fasta_file, output_dir)
        total = 0
        kept = 0
        translated_frames = 0
        has_translation = False
        
        # 先读取一次判断是否需要翻译，如果全是蛋白则无需创建子文件夹（虽然路径已经修改为根目录）
        # 但这里我们主要目的是为了逻辑统一：如果都是蛋白，直接复制或输出
        
        with open(fasta_file, 'r') as handle, open(output_file, 'w') as out:
            for record in SeqIO.parse(handle, "fasta"):
                total += 1
                seq = str(record.seq)
                if _looks_like_cds(seq):
                    has_translation = True
                    try:
                        translated = _six_frame_translate_all(seq, table=genetic_code)
                        for aa, frame in translated:
                            out.write(f">{record.id}|frame={frame}\n")
                            out.write(aa + "\n")
                            translated_frames += 1
                    except Exception:
                        out.write(f">{record.id}\n")
                        out.write(seq + "\n")
                        kept += 1
                else:
                    out.write(f">{record.id}\n")
                    out.write(seq + "\n")
                    kept += 1
        
        print(f"共读取 {total} 条序列，其中保留蛋白 {kept} 条，6转翻译生成帧序列 {translated_frames} 条")
        print(f"已输出到: {output_file}")
        
        return output_file
    except Exception as e:
        print(f"6转翻译输出蛋白FASTA失败: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='生成包含TF分类信息的FASTA文件或进行6转翻译')
    parser.add_argument('-i', '--input', required=True, help='输入FASTA文件路径')
    parser.add_argument('-o', '--output', help='输出FASTA文件路径（可选）')
    parser.add_argument('--classification', help='分类结果JSON文件路径（用于测试）')
    parser.add_argument('--translate-only', action='store_true', help='仅执行6转翻译并输出蛋白FASTA')
    parser.add_argument('--genetic-code', type=int, default=1, help='翻译使用的遗传密码表编号')
    parser.add_argument('--min-orf-aa', type=int, default=30, help='最小ORF氨基酸长度阈值')
    
    args = parser.parse_args()
    
    input_name = os.path.splitext(os.path.basename(args.input))[0]
    if not args.output:
        if args.translate_only:
            args.output = get_processed_fasta_path(args.input)
        else:
            output_dir = get_output_dir()
            args.output = os.path.join(output_dir, f"{input_name}_tf_classified.fasta")

    if args.translate_only:
        out = generate_protein_fasta_with_translation(
            args.input,
            output_file=args.output,
            output_dir=None,
            genetic_code=args.genetic_code,
            min_orf_aa=args.min_orf_aa,
        )
        if out:
            print(f"蛋白FASTA已保存到: {out}")
        else:
            print("6转翻译失败")
    else:
        if args.classification:
            try:
                with open(args.classification, 'r', encoding='utf-8') as f:
                    classification_result = json.load(f)
                print(f"从文件加载了 {len(classification_result)} 个分类结果")
            except Exception as e:
                print(f"读取分类结果文件时出错: {e}")
                classification_result = {}
        else:
            classification_result = {}
            print("未提供分类结果，将生成不包含TF分类信息的FASTA文件")
        success = format_tf_fasta_with_classification(args.input, classification_result, args.output)
        if success:
            print(f"FASTA文件已保存到: {args.output}")
        else:
            print("FASTA文件生成失败")

if __name__ == "__main__":
    main()
