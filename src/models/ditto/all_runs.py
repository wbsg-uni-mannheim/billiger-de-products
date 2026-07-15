import os 


sizes = ['small', 'medium', 'large'] 
difficulties = ['20cc80rnd', '50cc50rnd', '80cc20rnd'] 
unseens = ['000un'] 
#Adjust train_ditto_english.py to train_ditto for german dataset
for seed in range(3):
    for size in sizes: 
        for difficulty in difficulties: 
            for unseen in unseens:
                #check if file exists witin src/models/ditto/output_en where size, difficulty, unsee and seed are in the file name
                file_name = "final_%s_%s%s_lm=roberta_da=del_dk=None_su=False_size=None_id=%d_english.txt" % (size, difficulty, unseen, seed)
                file_path = "src/models/ditto/output_en/%s" % file_name
                print(file_path)
                if os.path.isfile(file_path):
                    print("File %s already exists, skipping..." % file_path)
                    continue
                cmd = """CUDA_VISIBLE_DEVICES=0 python src/models/ditto/train_ditto_english.py \
                        --task final_%s_%s%s \
                        --logdir src/models/ditto/results_en/ \
                        --run_id %d \
                        --batch_size 64 \
                        --max_len 256 \
                        --lr 5e-5 \
                        --n_epochs 50 \
                        --finetuning \
                        --lm roberta \
                        --da del""" % (size, difficulty, unseen, seed)
                print(cmd)
                os.system(cmd)