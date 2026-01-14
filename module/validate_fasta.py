#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iTAK 2.0 FASTA文件验证模块
检查输入的FASTA文件格式是否正确，是否为蛋白序列，是否包含星号
"""

import os
import re
from pathlib import Path
from Bio import SeqIO

class FastaValidator:
    """FASTA文件验证器类"""
    
    def __init__(self):
        self.errors = []
        self.protein_count = 0
        
        # 标准氨基酸字符集
        self.valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
        # 扩展氨基酸字符集（包括模糊字符）
        self.extended_amino_acids = set('ACDEFGHIKLMNPQRSTVWYXBZJU')
        

    
    def validate_fasta_format(self, fasta_file):
        try:
            sequences = list(SeqIO.parse(fasta_file, "fasta"))
            if not sequences:
                self.errors.append("文件不包含任何FASTA序列")
                return False
            
            sequence_count = len(sequences)
            print(f"  [#] 找到 {sequence_count} 个序列")
            return True
            
        except Exception as e:
            self.errors.append(f"FASTA格式解析错误: {str(e)}")
            return False
    
    def validate_protein_sequences(self, fasta_file):

        try:
            sequences = list(SeqIO.parse(fasta_file, "fasta"))
            self.protein_count = 0
            
            for record in sequences:
                seq_str = str(record.seq).upper()
                
                # 检查是否包含星号
                if '*' in seq_str:
                    self.errors.append(f"序列 {record.id} 包含禁止字符: *")
                    continue
                
                # 检查是否为蛋白质序列
                seq_chars = set(seq_str)
                
                # 计算有效氨基酸比例
                valid_chars = seq_chars & self.extended_amino_acids
                total_chars = len(seq_chars)
                
                if total_chars == 0:
                    self.errors.append(f"序列 {record.id} 为空序列")
                    continue
                
                valid_ratio = len(valid_chars) / total_chars
                
                if valid_ratio < 0.8:  # 至少80%的字符应该是有效的氨基酸
                    invalid_chars = seq_chars - self.extended_amino_acids
                    self.errors.append(f"序列 {record.id} 包含非蛋白质字符: {', '.join(invalid_chars)}")
                    continue
                
                # 通过验证的蛋白质序列
                self.protein_count += 1
            
            if self.protein_count == 0:
                self.errors.append("没有找到有效的蛋白质序列")
                return False
            
            print(f"  [#] 找到 {self.protein_count} 个有效蛋白质序列")
            return True
            
        except Exception as e:
            self.errors.append(f"序列验证过程中出错: {str(e)}")
            return False

    def validate_sequences(self, sequences):
        try:
            self.protein_count = 0
            for item in sequences:
                seq_str = str(item if isinstance(item, str) else item.get('sequence', '')).upper()
                if '*' in seq_str:
                    self.errors.append("序列包含禁止字符: *")
                    continue
                seq_chars = set(seq_str)
                if len(seq_chars) == 0:
                    self.errors.append("空序列")
                    continue
                valid_chars = seq_chars & self.extended_amino_acids
                valid_ratio = len(valid_chars) / len(seq_chars)
                if valid_ratio < 0.8:
                    invalid_chars = seq_chars - self.extended_amino_acids
                    self.errors.append(f"序列包含非蛋白质字符: {', '.join(invalid_chars)}")
                    continue
                self.protein_count += 1
            if self.protein_count == 0:
                self.errors.append("没有找到有效的蛋白质序列")
                return False
            return True
        except Exception as e:
            self.errors.append(f"序列验证过程中出错: {str(e)}")
            return False
    

    
    def run_full_validation(self, fasta_file):

        print(f"[#] 开始验证FASTA文件: {fasta_file}")
        
        # 重置错误列表
        self.errors = []
        
        # 1. 基本格式验证
        if not self.validate_fasta_format(fasta_file):
            print("[警告] FASTA格式验证失败")
            return False
        
        # 2. 蛋白质序列验证（包含星号检查和蛋白质数量统计）
        if not self.validate_protein_sequences(fasta_file):
            print("[警告] 蛋白质序列验证失败")
            return False
        
        print("[#] FASTA文件验证通过")
        return True
    


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) != 2:
        print("用法: python validate_fasta.py <fasta_file>")
        sys.exit(1)
    
    fasta_file = sys.argv[1]
    validator = FastaValidator()
    success = validator.run_full_validation(fasta_file)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
