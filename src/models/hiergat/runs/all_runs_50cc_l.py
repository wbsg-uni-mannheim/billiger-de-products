import os 

sizes = ['large'] 
difficulties = ['50cc50rnd'] 
unseens = ['000un'] 

for seed in range(3):
    for size in sizes: 
        for difficulty in difficulties: 
            for unseen in unseens: 
                file_name = "final_%s_%s%s_lr=5e-06_id=%d_batch=16_lm=roberta_adjusted.txt" % (size, difficulty, unseen, seed)
                file_path = "src/models/hiergat/output/%s" % file_name
                print(file_path)
                if os.path.isfile(file_path):
                    print("File %s already exists, skipping..." % file_path)
                    continue
                cmd = """CUDA_VISIBLE_DEVICES=0 python src/models/hiergat/train.py \
                        --task final_%s_%s%s \
                        --run_id %d \
                        --batch_size 16 \
                        --max_len 256 \
                        --lr 5e-6 \
                        --n_epochs 50 \
                        --finetuning \
                        --split \
                        --lm roberta""" % (size, difficulty, unseen, seed)
                print(cmd)
                os.system(cmd)
               