#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess
from pathlib import Path
from Bio import SeqIO
import csv
from datetime import datetime
import time
import json
import shutil
import tarfile

# 导入依赖检测模块
try:
    from module.check_dependencies import DependencyChecker
except ImportError:
    print("警告: 无法导入依赖检测模块，将跳过依赖检查")
    DependencyChecker = None

# 导入FASTA验证模块
try:
    from module.validate_fasta import FastaValidator
except ImportError:
    print("警告: 无法导入FASTA验证模块，将跳过FASTA验证")
    FastaValidator = None

# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent.absolute()
PREDICT_SCRIPT = SCRIPT_DIR / "pre_model" / "predict.py"
MODULE_DIR = SCRIPT_DIR / "module"

# DB相关路径和自动解压功能
DB_DIR = SCRIPT_DIR / "db"
DB_ARCHIVE = SCRIPT_DIR / "db.tar.gz"
INTERPROSCAN_SCRIPT = DB_DIR / "interproscan" / "interproscan.sh"

def ensure_db_extracted():
    """
    确保db文件夹已解压。如果db文件夹不存在，尝试从压缩包解压或从GitHub下载。
    使用 module/db_manager.py 实现。
    
    Returns:
        bool: 如果db文件夹可用返回True，否则返回False
    """
    try:
        import importlib.util
        db_manager_path = MODULE_DIR / "db_manager.py"
        
        if not db_manager_path.exists():
            print(f"错误: db_manager模块不存在: {db_manager_path}")
            # 回退到旧逻辑 (仅检查存在性)
            if DB_DIR.exists() and DB_DIR.is_dir():
                return True
            return False
            
        spec = importlib.util.spec_from_file_location("db_manager", db_manager_path)
        db_manager = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(db_manager)
        
        return db_manager.setup_db(SCRIPT_DIR)
            
    except Exception as e:
        print(f"检查/准备db文件夹时出错: {e}")
        return False


# 基础工具函数
def call_module_function(module_name, function_name, *args, **kwargs):

    try:
        # 构建模块路径
        module_path = MODULE_DIR / f"{module_name}.py"
        
        if not module_path.exists():
            raise FileNotFoundError(f"模块文件不存在: {module_path}")
        
        # 动态导入模块
        import importlib.util
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 获取函数
        if not hasattr(module, function_name):
            raise AttributeError(f"模块 {module_name} 中不存在函数 {function_name}")
        
        func = getattr(module, function_name)
        
        # 调用函数
        return func(*args, **kwargs)
        
    except Exception as e:
        print(f"调用模块函数时出错: {e}")
        return None
#计时器
def format_duration(seconds):

    if seconds < 60:
        return f"{seconds:.2f} 秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes} 分 {remaining_seconds:.2f} 秒"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours} 小时 {remaining_minutes} 分 {remaining_seconds:.2f} 秒"

def setup_project_output(fasta_file, output=None):

    if output:
        project_output = Path(output)
    else:
        # 使用默认输出目录结构
        fasta_path = Path(fasta_file)
        base_name = fasta_path.stem
        output_base = SCRIPT_DIR / "output"
        
        # 创建唯一的输出目录
        counter = 0
        while True:
            if counter == 0:
                project_output = output_base / base_name
            else:
                project_output = output_base / f"{base_name}_{counter}"
            
            if not project_output.exists():
                break
            counter += 1
    
    # 创建输出目录
    project_output.mkdir(parents=True, exist_ok=True)
    return project_output

# 模块分析功能
def run_analysis_modules(project_output, fasta_file, use_predicted=True, debug=False, score_threshold=1.0, classification_mode='specific'):
    """
    运行分析模块进行转录因子分类
    
    Args:
        project_output (Path): 项目输出目录
        fasta_file (str): 输入的FASTA文件路径
        use_predicted (bool): 是否使用预测后的序列文件
        debug (bool): 是否启用调试模式，输出调试文件
        score_threshold (float): InterProScan结果的score阈值
        classification_mode (str): 分类模式 ('specific' 或 'score')
    
    Returns:
        bool: 是否成功完成分析
    """
    print(f"\n{'='*50}")
    print("开始运行分析模块...")
    print(f"{'='*50}")
    
    try:
        # 确定要使用的FASTA文件
        if use_predicted:
            # 使用预测后的TF序列文件
            tf_fasta_file = project_output / "protein_model_preclassification" / f"{Path(fasta_file).stem}_tf_sequences.fasta"
            if not tf_fasta_file.exists():
                print(f"警告: 预测的TF序列文件不存在: {tf_fasta_file}")
                print("将使用原始输入文件进行分析")
                analysis_fasta = fasta_file
            else:
                analysis_fasta = str(tf_fasta_file)
        else:
            try:
                import importlib.util
                get_fasta_path = MODULE_DIR / "get_fasta.py"
                spec = importlib.util.spec_from_file_location("get_fasta", get_fasta_path)
                get_fasta = module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                processed_path = module.get_processed_fasta_path(fasta_file, project_output)
                if not Path(processed_path).exists():
                    module.generate_protein_fasta_with_translation(fasta_file, output_dir=project_output)
                analysis_fasta = processed_path
            except Exception:
                analysis_fasta = fasta_file
        
        # 确定其他必需的文件路径
        # InterProScan生成的文件名包含完整的输入文件名（包括扩展名）
        input_filename = Path(analysis_fasta).name
        ipr_file = project_output / "InterproScan" / f"{input_filename}.json"
        hmmscan_file = project_output / "hmmscan" / "result.tbl"
        # 确保db文件夹已解压
        if not ensure_db_extracted():
            print("错误: 无法获取db文件夹")
            return False
            
        rule_file = SCRIPT_DIR / "rule.txt"

        # 在测试模式下，ipr_file可能需要根据analysis_fasta的文件名进行调整
        # 因为run_test_mode中复制文件时可能使用了不同的文件名
        if not ipr_file.exists():
            # 尝试查找其他可能的JSON文件
            # 1. 尝试直接使用原始FASTA文件名的JSON (如果analysis_fasta是处理后的)
            original_json_file = project_output / "InterproScan" / f"{Path(fasta_file).name}.json"
            if original_json_file.exists():
                print(f"警告: 未找到 {ipr_file.name}，使用 {original_json_file.name} 替代")
                ipr_file = original_json_file
            else:
                # 2. 尝试查找目录下的唯一JSON文件
                json_files = list((project_output / "InterproScan").glob("*.json"))
                if len(json_files) == 1:
                    print(f"警告: 未找到 {ipr_file.name}，使用目录中唯一的JSON文件: {json_files[0].name}")
                    ipr_file = json_files[0]
        
        # 检查必需文件是否存在
        missing_files = []
        if not ipr_file.exists():
            missing_files.append(str(ipr_file))
        if not hmmscan_file.exists():
            missing_files.append(str(hmmscan_file))
        if not rule_file.exists():
            missing_files.append(str(rule_file))
        
        if missing_files:
            print("错误: 以下必需文件不存在:")
            for file in missing_files:
                print(f"  - {file}")
            return False
        
        print(f"使用的文件:")
        print(f"  FASTA文件: {analysis_fasta}")
        print(f"  IPR文件: {ipr_file}")
        print(f"  hmmscan文件: {hmmscan_file}")
        print(f"  规则文件: {rule_file}")
        
        # 创建result输出目录（在项目文件夹下）
        result_dir = project_output / "result"
        result_dir.mkdir(exist_ok=True)
        print(f"结果将输出到: {result_dir}")
        
        # 步骤0: 运行getrule模块，将规则文件JSON输出到项目文件夹根目录
        print("\n步骤0: 运行getrule模块...")
        getrule_success = run_getrule_module(str(rule_file), str(project_output), debug=debug)
        if not getrule_success:
            print("getrule模块运行失败")
            return False
        
        # 步骤1: 运行jsonbuild模块（支持内存数据传递）
        print("\n步骤1: 运行jsonbuild模块...")
        
        jsonbuild_result = run_jsonbuild_module(str(ipr_file), result_dir, debug=debug, score_threshold=score_threshold)
        
        # 检查jsonbuild结果
        if isinstance(jsonbuild_result, tuple) and len(jsonbuild_result) == 3:
            jsonbuild_success, filtered_data, raw_data = jsonbuild_result
            if not jsonbuild_success:
                print("jsonbuild模块运行失败")
                return False
            print("jsonbuild模块成功返回内存数据")
        else:
            # 兼容旧的返回格式
            jsonbuild_success = jsonbuild_result
            if not jsonbuild_success:
                print("jsonbuild模块运行失败")
                return False
            filtered_data = None
            raw_data = None
            print("jsonbuild模块使用文件传递方式")
        
        # 步骤2: 运行specpfam模块
                # 步骤2: 运行specpfam模块
        print("\n步骤2: 运行selfbuild_hmm模块...")
        specpfam_result = run_specpfam_module(str(hmmscan_file), result_dir, debug=debug)
        
        # 处理specpfam模块的返回结果
        if isinstance(specpfam_result, tuple):
            specpfam_success, spec_data = specpfam_result
        else:
            # 兼容旧的返回格式
            specpfam_success = specpfam_result
            spec_data = None
            
        if not specpfam_success:
            print("selfbuild_hmm模块运行失败")
            return False
            
        if spec_data is not None:
            print("成功获取selfbuild_hmm数据用于分类")
        else:
            print("未获取到selfbuild_hmm数据，将仅使用IPR数据进行分类")
        
        # 步骤3: 运行class_tf模块（支持内存数据传递）
        print("\n步骤3: 运行class_tf模块...")
        class_tf_success = run_class_tf_module(
            analysis_fasta, 
            str(rule_file), 
            result_dir, 
            debug=debug,
            filtered_data=filtered_data,
            spec_data=spec_data,
            classification_mode=classification_mode
        )
        if not class_tf_success:
            print("class_tf模块运行失败")
            return False
        
        print(f"\n{'='*50}")
        print("所有分析模块运行完成!")
        print(f"结果已保存到: {result_dir}")
        print(f"{'='*50}")
        
        return True
        
    except Exception as e:
        print(f"运行分析模块时出错: {e}")
        return False

def run_getrule_module(rule_file, output_dir, debug=False):
    try:
        import importlib.util
        get_rule_path = MODULE_DIR / "get_rule.py"
        spec = importlib.util.spec_from_file_location("get_rule", get_rule_path)
        get_rule = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(get_rule)
        rules = get_rule.parse_rule_file(rule_file)
        if debug:
            output_file = os.path.join(output_dir, 'getrule.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(rules, f, indent=2, ensure_ascii=False)
            print(f"规则已成功解析并保存到 {output_file}")
        print("get_rule模块运行成功")
        return True
    except Exception as e:
        print(f"运行get_rule模块时出错: {e}")
        return False
#从interproscan结果json文件中提取信息
def run_jsonbuild_module(ipr_file, result_dir, debug=False, score_threshold=1.0):

    try:
        # 首先尝试直接调用模块函数（内存传递）
        try:
            # 动态导入get_json模块
            import importlib.util
            get_json_path = MODULE_DIR / "get_json.py"
            spec = importlib.util.spec_from_file_location("get_json", get_json_path)
            get_json = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(get_json)
            
            print("使用内存数据传递方式运行get_json模块...")
            
            # 调用process_data函数
            filtered_result, new_result = get_json.process_data(
                input_file=str(ipr_file),
                output_dir=str(result_dir),  # 总是保存文件，因为classification模块需要
                debug=debug,
                score_threshold=score_threshold
            )
            
            if filtered_result is not None:
                print("get_json模块运行成功（内存传递）")
                return True, filtered_result, new_result
            else:
                print("get_json模块处理失败")
                return False, None, None
                
        except Exception as e:
            print(f"内存传递方式失败，回退到命令行方式: {e}")
            
            # 回退到原来的命令行方式
            get_json_script = MODULE_DIR / "get_json.py"
            
            # 构建命令行参数
            cmd = [
                sys.executable,
                str(get_json_script),
                "-i", str(ipr_file),
                "-o", str(result_dir),
                "--score", str(score_threshold)
            ]
            
            # 如果启用debug模式，添加--debug参数
            if debug:
                cmd.append("--debug")
            
            print(f"运行命令: {' '.join(cmd)}")
            
            # 执行get_json模块
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPT_DIR)
            
            if result.returncode == 0:
                print("get_json模块运行成功（命令行方式）")
                if result.stdout:
                    print(result.stdout.strip())
                return True, None, None  # 命令行方式不返回数据
            else:
                print("get_json模块运行失败")
                if result.stderr:
                    print(f"错误信息: {result.stderr}")
                if result.stdout:
                    print(f"输出信息: {result.stdout}")
                return False, None, None
            
    except Exception as e:
        print(f"运行get_json模块时出错: {e}")
        return False, None, None

def run_specpfam_module(hmmscan_file, result_dir, debug=False):
    """运行selfbuild_hmm模块"""
    try:
        # 直接导入并调用selfbuild_hmm模块
        import importlib.util
        selfbuild_hmm_path = MODULE_DIR / "selfbuild_hmm.py"
        spec = importlib.util.spec_from_file_location("selfbuild_hmm", selfbuild_hmm_path)
        selfbuild_hmm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(selfbuild_hmm)
        
        # 解析hmmscan结果
        result = selfbuild_hmm.parse_pfam_spec(hmmscan_file)
        
        # 只在debug模式下保存pfamspec.json文件
        if debug:
            output_file = os.path.join(result_dir, 'pfamspec.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"selfbuild_hmm模块处理完成，结果保存到: {output_file}")
        else:
            print("selfbuild_hmm模块处理完成（debug模式关闭，未保存pfamspec.json）")
        
        print("selfbuild_hmm模块运行成功")
        return True, result  # 返回成功状态和解析结果
            
    except Exception as e:
        print(f"运行selfbuild_hmm模块时出错: {e}")
        return False

#运行分类功能
def run_class_tf_module(fasta_file, rule_file, result_dir, debug=False, filtered_data=None, spec_data=None, classification_mode='specific'):

    try:
        # 首先尝试直接调用模块函数（内存传递）
        try:
            # 动态导入classification模块
            import importlib.util
            classification_path = MODULE_DIR / "classification.py"
            spec = importlib.util.spec_from_file_location("classification", classification_path)
            classification = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(classification)
            
            print("使用内存数据传递方式运行classification模块...")
            
            # 调用process_with_data函数
            classification_result = classification.process_with_data(
                result_directory=str(result_dir),
                rule_file=str(rule_file),
                filtered_data=filtered_data,
                spec_data=spec_data,
                debug=debug,
                mode=classification_mode
            )
            
            if classification_result is not None:
                # 保存结果文件
                tbl_path = os.path.join(result_dir, 'match_tbl.txt')
                with open(tbl_path, 'w', encoding='utf-8') as f:
                    for gene_id, data in classification_result.items():
                        desc_str = ';'.join(data['desc']) if data['desc'] else 'NA'
                        line = f"{gene_id}\t{data['name']}\t{data['family']}\t{data['type']}\t{desc_str}\t{data['other_family']}\n"
                        f.write(line)
                
                # 生成分类后的序列文件
                try:
                    # 动态导入get_fasta模块
                    get_fasta_path = MODULE_DIR / "get_fasta.py"
                    spec = importlib.util.spec_from_file_location("get_fasta", get_fasta_path)
                    get_fasta = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(get_fasta)
                    
                    if classification_result:
                        classified_fasta_path = get_fasta.generate_classified_fasta(
                            fasta_file, 
                            classification_result, 
                            output_dir=result_dir
                        )
                        if classified_fasta_path:
                            print(f"分类序列文件已生成: {classified_fasta_path}")
                        else:
                            print("分类序列文件生成失败")
                    else:
                        print("没有分类结果，跳过序列文件生成")
                except Exception as e:
                    print(f"生成分类序列文件时出错: {e}")
                
                # 只在debug模式下保存match.json文件
                if debug:
                    json_path = os.path.join(result_dir, 'match.json')
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(classification_result, f, indent=2, ensure_ascii=False)
                    
                    print(f"classification模块处理完成（内存传递）")
                    print(f"结果已保存到:")
                    print(f"  JSON格式: {json_path}")
                    print(f"  表格格式: {tbl_path}")
                else:
                    print(f"classification模块处理完成（内存传递）")
                    print(f"结果已保存到:")
                    print(f"  表格格式: {tbl_path}")
                
                return True
            else:
                print("classification模块处理失败")
                return False
                
        except Exception as e:
            print(f"运行classification模块时出错: {e}")
            return False
            
    except Exception as e:
        print(f"运行classification模块时出错: {e}")
        return False

# ============================================================================
# TF序列提取功能
# ============================================================================

def extract_tf_sequences_from_memory(fasta_file, tf_headers, output_dir):
    """
    从内存中的TF列表提取TF序列
    
    Args:
        fasta_file (str): 原始FASTA文件路径
        tf_headers (list): TF序列的header列表
        output_dir (str): 输出目录
    
    Returns:
        str: 输出FASTA文件路径，如果失败返回None
    """
    try:
        print(f"=== 提取TF序列（内存传递） ===")
        print(f"符合阈值的TF数量: {len(tf_headers)}")
        
        if not tf_headers:
            print("没有符合阈值的TF序列需要提取")
            return None
        
        # 创建输出文件路径
        fasta_basename = Path(fasta_file).stem
        output_fasta = Path(output_dir) / f"{fasta_basename}_tf_sequences.fasta"
        
        # 提取序列
        extracted_count = 0
        with open(output_fasta, 'w') as output_handle:
            with open(fasta_file, 'r') as input_handle:
                for record in SeqIO.parse(input_handle, "fasta"):
                    # 检查序列ID或描述是否在TF列表中
                    if record.id in tf_headers or record.description in tf_headers:
                        SeqIO.write(record, output_handle, "fasta")
                        extracted_count += 1
        
        print(f"成功提取 {extracted_count} 个TF序列")
        
        if extracted_count > 0:
            print(f"TF序列已保存到: {output_fasta}")
            return str(output_fasta)
        else:
            return None
            
    except Exception as e:
        print(f"提取TF序列时出错: {e}")
        return None
#提取序列
def extract_tf_sequences_from_csv(fasta_file, csv_file, output_dir, threshold):
    try:
        # 读取CSV文件获取TF序列的header
        tf_headers = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Predicted_Class'] == 'TF' and float(row['TF_Probability']) >= threshold:
                    tf_headers.append(row['Header'])
        
        print(f"=== 提取TF序列 ===")
        print(f"符合阈值的TF数量: {len(tf_headers)}")
        
        if not tf_headers:
            print("没有符合阈值的TF序列需要提取")
            return None
        
        # 创建输出文件路径
        fasta_basename = Path(fasta_file).stem
        output_fasta = Path(output_dir) / f"{fasta_basename}_tf_sequences.fasta"
        
        # 提取序列
        extracted_count = 0
        with open(output_fasta, 'w') as output_handle:
            with open(fasta_file, 'r') as input_handle:
                for record in SeqIO.parse(input_handle, "fasta"):
                    # 检查序列ID或描述是否在TF列表中
                    if record.id in tf_headers or record.description in tf_headers:
                        SeqIO.write(record, output_handle, "fasta")
                        extracted_count += 1
        
        print(f"成功提取 {extracted_count} 个TF序列")
        
        if extracted_count > 0:
            print(f"TF序列已保存到: {output_fasta}")
            return str(output_fasta)
        else:
            return None
            
    except Exception as e:
        print(f"提取TF序列时出错: {e}")
        return None

# InterProScan功能

def run_interproscan(fasta_file, output_dir, appl_list=None):
    try:
        # 确保db文件夹已解压
        if not ensure_db_extracted():
            print("错误: 无法获取db文件夹")
            return False
            
        # 创建ipr输出目录
        ipr_output_dir = Path(output_dir) / "InterproScan"
        ipr_output_dir.mkdir(exist_ok=True)
        
        # 构建InterProScan命令
        interproscan_script = str(INTERPROSCAN_SCRIPT)
        cmd = [interproscan_script, '-i', fasta_file, '-f', 'json', '-d', str(ipr_output_dir)]
        
        if appl_list:
            cmd.extend(['-appl', appl_list])
        
        print(f"运行命令: {' '.join(cmd)}")
        
        # 运行命令
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("InterProScan分析完成")
            return True
        else:
            print(f"InterProScan分析失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"运行InterProScan时出错: {e}")
        return False

#hmmscan运行
def run_hmmscan(fasta_file, output_dir):
    # 确保db文件夹已解压
    if not ensure_db_extracted():
        print("错误: 无法获取db文件夹")
        return False
        
    # 设置hmmscan相关路径
    hmm_db = DB_DIR / "self_build_hmm" / "self_build.hmm"
    
    if not hmm_db.exists():
        print(f"错误: HMM数据库文件不存在: {hmm_db}")
        return False
    
    if not Path(fasta_file).exists():
        print(f"错误: FASTA文件不存在: {fasta_file}")
        return False
    
    # 优先使用内部hmmscan (位于 db/interproscan/bin/hmmer/hmmer3/hmmscan)
    # 路径: db/interproscan/bin/hmmer/hmmer3/hmmscan
    internal_hmmscan = DB_DIR / "interproscan" / "bin" / "hmmer" / "hmmer3" / "hmmscan"
    
    if internal_hmmscan.exists() and os.access(internal_hmmscan, os.X_OK):
        hmmscan_executable = str(internal_hmmscan)
        print(f"使用内置hmmscan: {hmmscan_executable}")
    else:
        # 回退到系统hmmscan
        hmmscan_executable = "hmmscan"
        if shutil.which("hmmscan") is None:
             print("错误: 找不到hmmscan可执行文件（内置或系统路径均未找到）")
             return False
        print(f"使用系统hmmscan: {hmmscan_executable}")

    try:
        # 创建hmmscan输出目录
        hmmscan_output_dir = Path(output_dir) / "hmmscan"
        hmmscan_output_dir.mkdir(exist_ok=True)
        
        # 设置输出文件路径
        output_file = hmmscan_output_dir / "result.tbl"
        
        # 构建hmmscan命令
        cmd = [
            hmmscan_executable,
            "--tblout", str(output_file),
            "--noali",
            str(hmm_db),
            str(fasta_file)
        ]
        
        print(f"运行hmmscan命令: {' '.join(cmd)}")
        
        # 运行hmmscan
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"hmmscan运行成功")
        print(f"结果已保存到: {output_file}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"hmmscan运行失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    except Exception as e:
        print(f"运行hmmscan时发生错误: {e}")
        return False

# ============================================================================
# 主要功能函数
# ============================================================================
#完整运行流程
def predict_transcription_factors(threshold, fasta_file, output=None, extract_sequences=True, 
                                run_interproscan_analysis=True, run_hmmscan_analysis=True, 
                                appl_list=None, debug=False, score_threshold=1.0, classification_mode='specific',
                                predict_mode='fast'):
    # 记录预测开始时间
    predict_start_time = time.time()
    
    try:
        # 输入检查
        if not Path(fasta_file).exists():
            print(f"错误: 输入文件不存在: {fasta_file}")
            return False
        
        # 设置项目输出目录
        project_output = setup_project_output(fasta_file, output)
        
        print(f"\n=== 开始转录因子预测 ===\n")
        print(f"输入文件: {fasta_file}")
        print(f"输出目录: {project_output}")
        print(f"预测阈值: {threshold}")
        print(f"预测模式: {predict_mode}")
        
        # 构建预测命令
        if not PREDICT_SCRIPT.exists():
            print(f"错误: 预测脚本不存在: {PREDICT_SCRIPT}")
            return False
        
        # 生成输出文件名
        fasta_basename = Path(fasta_file).stem
        
        # 创建预分类结果目录
        preclass_dir = project_output / "protein_model_preclassification"
        preclass_dir.mkdir(exist_ok=True)
        
        output_file = preclass_dir / f"{fasta_basename}_prediction.csv"
        
        cmd = [
            "python", str(PREDICT_SCRIPT),
            "--fasta", fasta_file,
            "--threshold", str(threshold),
            "--output", str(output_file),
            "--project-output", str(project_output),
            "--mode", predict_mode
        ]
        
        # 在debug模式下添加debug参数
        if debug:
            cmd.append("--debug")
        
        # 非debug模式不输出TF列表到终端，依赖预测脚本写出的FASTA
        
        print(f"执行命令: {' '.join(cmd)}")
        
        # 执行预测
        print("1. 运行转录因子预测...")
        step_start_time = time.time()
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPT_DIR)
        
        step_end_time = time.time()
        step_duration = step_end_time - step_start_time
        
        if result.returncode != 0:
            print(f"预测失败: {result.stderr}")
            return False
        
        print(f"预测完成 (耗时: {format_duration(step_duration)})")
        
        # 显示预测脚本的输出信息（包含序列拆分统计等）
        if result.stdout:
            print(result.stdout)
        
        # 提取预测的TF序列
        tf_fasta = None
        if extract_sequences:
            print("\n2. 提取预测的转录因子序列...")
            step_start_time = time.time()
            
            # 创建fasta文件夹
            fasta_output_dir = project_output / "protein_model_preclassification"
            fasta_output_dir.mkdir(exist_ok=True)
            # 使用处理后的蛋白FASTA作为提取源，避免原始FASTA无法匹配frame后缀
            try:
                import importlib.util
                get_fasta_path = MODULE_DIR / "get_fasta.py"
                spec = importlib.util.spec_from_file_location("get_fasta", get_fasta_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                processed_fasta = module.get_processed_fasta_path(fasta_file, project_output)
                if not processed_fasta.exists():
                    module.generate_protein_fasta_with_translation(fasta_file, output_dir=project_output)
            except Exception:
                processed_fasta = Path(fasta_file)
            
            if debug:
                # Debug模式：从CSV文件提取
                prediction_file = output_file
                if not prediction_file.exists():
                    print(f"警告: 未找到预测结果文件: {prediction_file}")
                    return False
                
                print(f"使用预测结果文件: {prediction_file}")
                tf_fasta = extract_tf_sequences_from_csv(str(processed_fasta), str(prediction_file), str(fasta_output_dir), threshold)
            else:
                # 非Debug模式：直接使用预测脚本写出的TF FASTA
                tf_fasta_path = fasta_output_dir / f"{fasta_basename}_tf_sequences.fasta"
                if tf_fasta_path.exists():
                    tf_fasta = str(tf_fasta_path)
                    print(f"已生成TF序列FASTA: {tf_fasta}")
                else:
                    print("未找到预测脚本写出的TF FASTA")
                    return False
            
            if tf_fasta is None:
                print("序列提取失败")
                return False
            
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            print(f"TF序列已提取到: {tf_fasta} (耗时: {format_duration(step_duration)})")
        
        # 运行InterProScan分析
        if run_interproscan_analysis and tf_fasta:
            print("\n3. 运行InterProScan分析...")
            step_start_time = time.time()
            
            interproscan_success = run_interproscan(tf_fasta, str(project_output), appl_list)
            
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            
            if interproscan_success:
                print(f"InterProScan分析完成 (耗时: {format_duration(step_duration)})")
            else:
                print("InterProScan分析失败")
        
        # 运行hmmscan分析
        if run_hmmscan_analysis and tf_fasta:
            print("\n4. 运行hmmscan分析...")
            step_start_time = time.time()
            
            hmmscan_success = run_hmmscan(tf_fasta, str(project_output))
            
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            
            if hmmscan_success:
                print(f"hmmscan分析完成 (耗时: {format_duration(step_duration)})")
            else:
                print("hmmscan分析失败")
        
        # 运行分析模块进行分类
        if run_interproscan_analysis and run_hmmscan_analysis and tf_fasta:
            print("\n5. 运行分析模块进行转录因子分类...")
            step_start_time = time.time()
            
            analysis_success = run_analysis_modules(project_output, fasta_file, use_predicted=True, debug=debug, score_threshold=score_threshold, classification_mode=classification_mode)
            
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            
            if analysis_success:
                print(f"分析模块运行完成 (耗时: {format_duration(step_duration)})")
            else:
                print("分析模块运行失败")
        
        # 计算总预测时间
        predict_end_time = time.time()
        total_predict_duration = predict_end_time - predict_start_time
        
        print("\n=== 转录因子预测完成 ===")
        print(f"总预测时间: {format_duration(total_predict_duration)}")
        print()
        return True
        
    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        return False
#不使用预测功能
def analyze_sequences_directly(fasta_file, output=None, appl_list=None, debug=False, score_threshold=1.0, classification_mode='specific'):
    # 记录分析开始时间
    analysis_start_time = time.time()
    
    try:
        # 输入检查
        if not Path(fasta_file).exists():
            print(f"错误: 输入文件不存在: {fasta_file}")
            return False
        
        # 设置项目输出目录
        project_output = setup_project_output(fasta_file, output)
        
        print(f"\n=== 开始序列分析 ===\n")
        print(f"输入文件: {fasta_file}")
        print(f"输出目录: {project_output}")
        
        try:
            import importlib.util
            get_fasta_path = MODULE_DIR / "get_fasta.py"
            spec = importlib.util.spec_from_file_location("get_fasta", get_fasta_path)
            get_fasta = module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            processed_fasta = module.get_processed_fasta_path(fasta_file, project_output)
            if not Path(processed_fasta).exists():
                module.generate_protein_fasta_with_translation(fasta_file, output_dir=project_output)
        except Exception:
            processed_fasta = fasta_file
        # 运行InterProScan分析
        print("1. 运行InterProScan分析...")
        step_start_time = time.time()
        interproscan_success = run_interproscan(processed_fasta, str(project_output), appl_list)
        
        step_end_time = time.time()
        step_duration = step_end_time - step_start_time
        
        if interproscan_success:
            print(f"InterProScan分析完成 (耗时: {format_duration(step_duration)})")
        else:
            print("InterProScan分析失败")
        
        # 运行hmmscan分析
        print("\n2. 运行hmmscan分析...")
        step_start_time = time.time()
        
        hmmscan_success = run_hmmscan(processed_fasta, str(project_output))
        
        step_end_time = time.time()
        step_duration = step_end_time - step_start_time
        
        if hmmscan_success:
            print(f"hmmscan分析完成 (耗时: {format_duration(step_duration)})")
        else:
            print("hmmscan分析失败")
        
        # 运行分析模块进行分类
        if interproscan_success and hmmscan_success:
            print("\n3. 运行分析模块进行转录因子分类...")
            step_start_time = time.time()
            
            analysis_success = run_analysis_modules(project_output, fasta_file, use_predicted=False, debug=debug, score_threshold=score_threshold, classification_mode=classification_mode)
            
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            
            if analysis_success:
                print(f"分析模块运行完成 (耗时: {format_duration(step_duration)})")
            else:
                print("分析模块运行失败")
        
        # 计算总分析时间
        analysis_end_time = time.time()
        total_analysis_duration = analysis_end_time - analysis_start_time
        
        print("\n=== 序列分析完成 ===")
        print(f"总分析时间: {format_duration(total_analysis_duration)}")
        print()
        return True
        
    except Exception as e:
        print(f"分析过程中发生错误: {e}")
        return False

# -test模式，使用使用现有的结果进行运行，测试分类的功能
def run_test_mode(fasta_file, json_file, spechmm_file, output=None, debug=False, score_threshold=1.0, classification_mode='specific'):
    print(f"\n{'='*50}")
    print("启动测试模式")
    print("跳过InterProScan和hmmscan步骤，直接使用提供的文件进行分类验证")
    print(f"{'='*50}")
    
    try:
        # 设置项目输出目录
        project_output = setup_project_output(fasta_file, output)
        print(f"项目输出目录: {project_output}")
        
        # 创建必要的子目录结构
        ipr_dir = project_output / "InterproScan"
        hmmscan_dir = project_output / "hmmscan"
        result_dir = project_output / "result"
        
        ipr_dir.mkdir(exist_ok=True)
        hmmscan_dir.mkdir(exist_ok=True)
        result_dir.mkdir(exist_ok=True)
        
        print(f"\n创建项目目录结构:")
        print(f"  IPR目录: {ipr_dir}")
        print(f"  hmmscan目录: {hmmscan_dir}")
        print(f"  结果目录: {result_dir}")
        
        # 复制提供的文件到项目目录中
        input_filename = Path(fasta_file).name
        target_json_file = ipr_dir / f"{input_filename}.json"
        target_hmmscan_file = hmmscan_dir / "result.tbl"
        
        print(f"\n复制输入文件到项目目录:")
        print(f"  复制 {json_file} -> {target_json_file}")
        shutil.copy2(json_file, target_json_file)
        
        print(f"  复制 {spechmm_file} -> {target_hmmscan_file}")
        shutil.copy2(spechmm_file, target_hmmscan_file)
        
        # 验证复制的文件
        if not target_json_file.exists():
            print(f"错误: 复制JSON文件失败: {target_json_file}")
            return False
        
        if not target_hmmscan_file.exists():
            print(f"错误: 复制hmmscan文件失败: {target_hmmscan_file}")
            return False
        
        print(f"\n文件复制完成，开始运行分析模块...")
        
        # 记录分析开始时间
        analysis_start_time = time.time()
        
        # 运行分析模块（使用原始FASTA文件，不使用预测序列）
        analysis_success = run_analysis_modules(project_output, fasta_file, use_predicted=False, debug=debug, score_threshold=score_threshold, classification_mode=classification_mode)
        
        # 计算分析时间
        analysis_end_time = time.time()
        analysis_duration = analysis_end_time - analysis_start_time
        
        if analysis_success:
            print(f"\n{'='*50}")
            print("测试模式分析完成!")
            print(f"分析时间: {format_duration(analysis_duration)}")
            print(f"结果已保存到: {result_dir}")
            print(f"{'='*50}")
            return True
        else:
            print(f"\n{'='*50}")
            print("测试模式分析失败!")
            print(f"分析时间: {format_duration(analysis_duration)}")
            print(f"{'='*50}")
            return False
        
    except Exception as e:
        print(f"测试模式运行过程中发生错误: {e}")
        return False

# ============================================================================
# 参数验证函数
# ============================================================================

def validate_appl_list(appl_string):
    valid_apps = {'CDD', 'PANTHER', 'Pfam', 'PROSITEPATTERNS', 'PROSITEPROFILES', 'SMART', 'TIGRFAM'}
    
    if appl_string:
        apps = [app.strip() for app in appl_string.split(',')]
        invalid_apps = [app for app in apps if app not in valid_apps]
        
        if invalid_apps:
            raise argparse.ArgumentTypeError(f"无效的应用程序: {', '.join(invalid_apps)}. 允许的应用程序: {', '.join(sorted(valid_apps))}")
    
    return appl_string
#验证预测阈值的大小，其大小范围应该为在0-1之间
def validate_threshold(value):
    try:
        threshold = float(value)
        if not 0 <= threshold <= 1:
            raise argparse.ArgumentTypeError("阈值必须在0-1之间")
        return threshold
    except ValueError:
        raise argparse.ArgumentTypeError("阈值必须是数字")

# ============================================================================
# 主函数
# ============================================================================

def main():
    # 记录程序开始时间
    program_start_time = time.time()
    start_datetime = datetime.now()
    print(f"\n=== iTAK2 程序启动 ===")
    print(f"启动时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"="*50)
    
    parser = argparse.ArgumentParser(
        description="iTAK2- 转录因子预测和分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用预测功能（序列先通过模型预测，然后分析预测的TF序列）
  python itak2.0.py --predict -i input.fasta -t 0.1
  
  # 直接分析输入序列（跳过预测步骤）
  python itak2.0.py -i input.fasta
  
  # 测试模式（直接使用指定的文件进行分类验证）
  python itak2.0.py -test -i input.fasta -json ipr_result.json -spechmm hmmscan_result.tbl
  
  # 自定义InterProScan应用程序
  python itak2.0.py -i input.fasta --appl CDD,Pfam,SMART
  
  # 指定输出目录
  python itak2.0.py --predict -i input.fasta -t 0.2 -o /path/to/output
        """
    )
    
    # 主要参数
    parser.add_argument('-i', '--input', help='输入FASTA文件路径')
    parser.add_argument('-t', '--threshold', type=validate_threshold, default=0.1, 
                       help='预测阈值，范围0-1 (默认: 0.1)')
    parser.add_argument('-o', '--output', help='输出目录路径')
    
    # 功能选项
    parser.add_argument('--predict', action='store_true', 
                       help='启用预测功能（将序列输入到模型中进行预测，预测结果作为后续分析的输入）')
    # parser.add_argument('--extract-sequences', action='store_true', default=True, 
    #                    help='提取预测的TF序列（仅在使用--predict时有效）')
    
    # 测试模式参数
    parser.add_argument('-test', '--test-mode', action='store_true',
                       help='启用测试模式（跳过InterProScan和hmmscan，直接使用指定文件进行分类验证）')
    parser.add_argument('-json', '--json-file', 
                       help='测试模式下的InterProScan JSON结果文件路径（仅在-test模式下使用）')
    parser.add_argument('-spechmm', '--spechmm-file',
                       help='测试模式下的hmmscan结果文件路径（仅在-test模式下使用）')
    
    # InterProScan和hmmscan默认启用，但可以通过参数控制
    parser.add_argument('--appl', type=validate_appl_list,
                       default='CDD,PANTHER,Pfam,PROSITEPATTERNS,PROSITEPROFILES,SMART,TIGRFAM',
                       help='InterProScan应用程序列表，用逗号分隔。允许的数据库: CDD,PANTHER,Pfam,PROSITEPATTERNS,PROSITEPROFILES,SMART,TIGRFAM (默认: 全部)')
    
    # 调试模式参数
    parser.add_argument('--debug', action='store_true', default=False,
                       help='启用调试模式，输出中间调试文件（默认: 关闭）')
    
    # Score阈值参数
    parser.add_argument('--score', type=float, default=1.0,
                       help='InterProScan结果的score阈值，只有大于此值的结果才会被保留 (默认: 1.0)')
    
    # 分类模式参数
    parser.add_argument('--classification-mode', choices=['specific', 'score'], default='score',
                       help="分类模式: 'specific' (特异性优先) 或 'score' (得分优先，默认)")
    
    # 预测模式参数
    parser.add_argument('--predict-mode', choices=['fast', 'full'], default='fast',
                       help="序列预测拆分模式: 'fast' (快速模式，默认) 或 'full' (全覆盖模式)")

    # 依赖检测参数
    parser.add_argument('--check-deps', action='store_true',
                       help='仅检查依赖并退出，不运行主程序')
    parser.add_argument('--skip-deps-check', action='store_true',
                       help='跳过依赖检查，直接运行程序（不推荐）')
    
    args = parser.parse_args()
    
    # 如果只是检查依赖，运行检查后退出
    if args.check_deps:
        if DependencyChecker:
            checker = DependencyChecker()
            success = checker.run_full_check()
            sys.exit(0 if success else 1)
        else:
            print("错误: 依赖检测模块不可用")
            sys.exit(1)
    
    # 运行依赖检查（除非明确跳过）
    if not args.skip_deps_check and DependencyChecker:
        print(" 正在检查程序依赖...")
        checker = DependencyChecker()
        
        # 根据命令行参数决定是否检查预测依赖
        check_predict = args.predict
        dependencies_ok = checker.run_full_check(check_predict=check_predict)
        
        # 如果db缺失，run_full_check 会尝试下载并解压。
        # 如果下载成功，dependencies_ok 应该为 True（前提是其他依赖也满足）。
        # 如果下载/解压失败，dependencies_ok 为 False。
        
        if not dependencies_ok:
            # 检查是否是因为数据库下载成功但需要重启（这种情况通常不会发生，因为setup_db是同步的）
            # 但为了安全起见，我们再次检查一次db是否就绪
            if checker.ensure_db_extracted():
                 # 如果db现在好了，但刚才报错了，可能是有其他依赖缺失
                 # 重新运行检查以确认
                 print("\n[信息] 数据库已准备就绪，重新检查所有依赖...")
                 dependencies_ok = checker.run_full_check(check_predict=check_predict)

        if not dependencies_ok:
            print("\n[错误] 依赖检查失败！程序可能无法正常运行。")
            print("请安装缺失的依赖，或使用 --skip-deps-check 强制运行（不推荐）")
            print("使用 --check-deps 可以单独运行依赖检查")
            sys.exit(1)
        
        print("[成功] 依赖检查通过！\n")
    elif args.skip_deps_check:
        print("[警告] 已跳过依赖检查\n")
    
    # 验证输入文件参数（依赖检查模式下不需要）
    if not args.input:
        print("错误: 必须提供输入文件 (-i/--input)")
        sys.exit(1)
    
    # 验证输入文件存在
    if not Path(args.input).exists():
        print(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)
    
    # 验证FASTA文件格式
    if FastaValidator:
        print(" 正在验证FASTA文件格式...")
        validator = FastaValidator()
        is_valid = validator.run_full_validation(args.input)
        
        if not is_valid:
            print("[错误] FASTA文件验证失败！请检查上述错误信息。")
            sys.exit(1)
        
        print("[成功] FASTA文件格式验证通过！\n")
    else:
        print("[警告] FASTA验证模块不可用，跳过格式验证\n")
    
    # 验证测试模式参数
    if args.test_mode:
        # 测试模式下不能使用--predict
        if args.predict:
            print("错误: 测试模式(-test)下不能使用--predict参数")
            sys.exit(1)
        
        # 测试模式下必须提供json和spechmm文件
        if not args.json_file:
            print("错误: 测试模式下必须提供-json参数指定InterProScan JSON结果文件")
            sys.exit(1)
        
        if not args.spechmm_file:
            print("错误: 测试模式下必须提供-spechmm参数指定hmmscan结果文件")
            sys.exit(1)
        
        # 验证测试模式下的文件是否存在
        if not Path(args.json_file).exists():
            print(f"错误: InterProScan JSON文件不存在: {args.json_file}")
            sys.exit(1)
        
        if not Path(args.spechmm_file).exists():
            print(f"错误: hmmscan结果文件不存在: {args.spechmm_file}")
            sys.exit(1)
    
    # 验证非测试模式下不能使用测试模式专用参数
    if not args.test_mode:
        if args.json_file:
            print("错误: -json参数只能在测试模式(-test)下使用")
            sys.exit(1)
        
        if args.spechmm_file:
            print("错误: -spechmm参数只能在测试模式(-test)下使用")
            sys.exit(1)
    
    # 根据模式参数决定处理流程
    if args.test_mode:
        # 测试模式：直接使用提供的文件进行分类验证
        print("使用测试模式")
        success = run_test_mode(
            fasta_file=args.input,
            json_file=args.json_file,
            spechmm_file=args.spechmm_file,
            output=args.output,
            debug=args.debug,
            score_threshold=args.score,
            classification_mode=args.classification_mode
        )
    elif args.predict:
        # 使用预测功能 - 首先检查预测功能的依赖
        print(f"使用预测模式，阈值: {args.threshold}")
        
        # 检查预测功能的依赖
        if DependencyChecker:
            checker = DependencyChecker()
            prediction_ok, error_msg = checker.check_prediction_dependencies()
            
            if not prediction_ok:
                print(f"\n[错误] 预测功能依赖检查失败: {error_msg}")
                print("\n预测功能需要以下依赖:")
                print("  - PyTorch (深度学习框架)")
                print("  - 预测模型文件 (model.pth)")
                print("  - 预测脚本 (predict.py)")
                print("\n安装PyTorch:")
                print("  # CPU版本:")
                print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
                print("  # GPU版本 (CUDA 11.8):")
                print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                print("  # 更多版本请访问: https://pytorch.org/get-started/locally/")
                print("\n提示: 如果不需要预测功能，可以移除 --predict 参数直接运行分析。")
                sys.exit(1)
            
            print("[成功] 预测功能依赖检查通过！")
        
        success = predict_transcription_factors(
            threshold=args.threshold,
            fasta_file=args.input,
            output=args.output,
            extract_sequences=True,  # 强制为True，因为这是必需步骤
            run_interproscan_analysis=True,  # 默认启用
            run_hmmscan_analysis=True,       # 默认启用
            appl_list=args.appl,
            debug=args.debug,
            score_threshold=args.score,
            classification_mode=args.classification_mode,
            predict_mode=args.predict_mode
        )
    else:
        # 直接分析输入文件，跳过预测步骤
        print("直接分析模式（跳过预测步骤）")
        success = analyze_sequences_directly(
            fasta_file=args.input,
            output=args.output,
            appl_list=args.appl,
            debug=args.debug,
            score_threshold=args.score,
            classification_mode=args.classification_mode
        )
    
    if success:
        print("\n分析完成!")
    else:
        print("\n分析失败!")
        # 计算并显示运行时间（即使失败也显示）
        program_end_time = time.time()
        end_datetime = datetime.now()
        total_runtime = program_end_time - program_start_time
        
        print(f"\n=== iTAK 2.0 程序结束 ===")
        print(f"结束时间: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总运行时间: {format_duration(total_runtime)}")
        print(f"="*50)
        sys.exit(1)
    
    # 计算并显示总运行时间
    program_end_time = time.time()
    end_datetime = datetime.now()
    total_runtime = program_end_time - program_start_time
    
    print(f"\n=== iTAK 2.0 程序结束 ===")
    print(f"结束时间: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总运行时间: {format_duration(total_runtime)}")
    print(f"="*50)

if __name__ == '__main__':
    main()
