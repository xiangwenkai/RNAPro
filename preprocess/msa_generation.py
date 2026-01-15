import os
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import time
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Generate MSA")
    parser.add_argument("-i", "--input_file", type=int, help="input csv file path", default='/inspire/ssd/project/sais-bio/public/xiangwenkai/project/kaggle/input/stanford-rna-3d-folding-2/test_sequences.csv')
    args = parser.parse_args()
    
    workdir = "/inspire/ssd/project/sais-bio/public/xiangwenkai/project/kaggle"
    rna_database_dir = "/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/RNAPro/release_data/rna_db"
    df_test = pd.read_csv(f'{args.input_file}')
    print(df_test.columns)

    # 将字符串转换为 SeqRecord 对象
    os.makedirs(f"{workdir}/process/input_fasta", exist_ok=True)
    os.makedirs(f"{workdir}/process/msa_fasta", exist_ok=True)

    for rna_id, seq_str in zip(df_test['target_id'].tolist(), df_test['sequence'].tolist()):
        t0 = time.time()
        record = SeqRecord(
            Seq(seq_str),
            id=rna_id,
            description=''
        )
        # 写入文件
        if os.path.exists(f"{workdir}//process/input_fasta/{rna_id}.fasta") == False:
            with open(f"{workdir}//process/input_fasta/{rna_id}.fasta", "w") as output_handle:
                SeqIO.write(record, output_handle, "fasta")
        if os.path.exists(f"{workdir}/process/msa_fasta/{rna_id}.MSA.fasta") == False:
            os.system(f"nhmmer --cpu 64 --rna -A {workdir}/process/input_fasta/{rna_id}.sto {workdir}/process/input_fasta/{rna_id}.fasta {rna_database_dir}/rnacentral_active.fasta")
            # os.system(f"hhfilter -i {workdir}/process/input_fasta/{rna_id}.sto -o {workdir}/process/input_fasta/{rna_id}_filtered.sto -id 90 -cov 50")
            os.system(f"esl-reformat fasta {workdir}/process/input_fasta/{rna_id}.sto > {workdir}/process/input_fasta/{rna_id}_res.fasta")
            if os.path.getsize(f"{workdir}/process/input_fasta/{rna_id}_res.fasta") > 0:
                os.system(f"cat {workdir}/process/input_fasta/{rna_id}.fasta {workdir}/process/input_fasta/{rna_id}_res.fasta > {workdir}/process/msa_fasta/{rna_id}.MSA.fasta")
                print(f"pdb id: {rna_id}; rna length: {len(seq_str)}; processing time: {time.time() - t0}")
            else:
                os.system(f"cp {workdir}//process/input_fasta/{rna_id}.fasta {workdir}/process/msa_fasta/{rna_id}.MSA.fasta")
                print(f"pdb id: {rna_id}; rna length: {len(seq_str)}; processing time: {time.time() - t0}; NO MSA!!!!")

if __name__ == "__main__":
    main()