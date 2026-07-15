import pandas as pd
from sklearn.metrics import f1_score
import os
import glob
import re
from sklearn.metrics import precision_recall_fscore_support

CATEGORY_PATH = 'data/working/dedup_preprocessed_rev2_docs_since_2020_01_01_only_de_strict_only_long_name_only_mainentity_with_new_category.pkl.gz'

def categories(language, model, prompt):
    if language == "de":
        language_full = "german"
    elif language == "en":
        language_full = "english"
    else:
        raise ValueError("Language not supported. Use 'de' for German or 'en' for English.")
    
    predictions_path = f"src/models/gpt/reports_{language}/{model}/csv_results/*.csv"
    df_categories = pd.read_pickle(CATEGORY_PATH, compression='gzip')
    category_analysis = []
    for file in glob.glob(predictions_path):
        if prompt not in file:
            continue
        df_predictions = pd.read_csv(file)
        # create new id_left and id_right from pair id, which is id_left#id_right
        df_predictions[["id_left", "id_right"]] = df_predictions["Pair_ID"].str.split("#", expand=True)
        #turn all ids to int
        df_predictions["id_left"] = df_predictions["id_left"].astype(int)
        df_predictions["id_right"] = df_predictions["id_right"].astype(int)
        name = file.split('/')[-1].removesuffix(".csv")
        gs_match = re.search(r"products_\d+cc\d+_\d+un_batched_{}_{}".format(language_full, prompt), name)
        gs_name = gs_match.group()

        print("Processing Testset:", gs_name, " for model:", model)
        # Merge for id_left -> shop_cat_left 
        df_merged_left = pd.merge(df_predictions, df_categories[['id', 'top_category_mapped']], left_on='id_left', right_on='id', how='left', suffixes=('', '_left') ) 
        
        # Rename shop_cat column correctly 
        df_merged_left.rename(columns={'top_category_mapped': 'shop_cat_left'}, inplace=True) 
        
        # Merge for id_right -> shop_cat_right 
        df_merged_both_test = pd.merge( df_merged_left, df_categories[['id', 'top_category_mapped']], left_on='id_right', right_on='id', how='left', suffixes=('', '_right') ) 
        
        # Rename shop_cat column from right merge 
        df_merged_both_test.rename(columns={'top_category_mapped': 'shop_cat_right'}, inplace=True) 
        
        df_merged_both_test["category_pair"] = (
            df_merged_both_test[["shop_cat_left", "shop_cat_right"]]
                .astype(str)
                .apply(lambda x: " | ".join(sorted(x)), axis=1)
        )


        #get amount of items grouped by category from df_merged_both_test
        category_pair_counts = (
            df_merged_both_test["category_pair"]
                .value_counts(dropna=False)
                .to_dict()
        )

        cc = re.search(r"\d+cc\d+", gs_name).group()
        un = re.search(r"\d+un", gs_name).group()
        
        # rename Match column to match and turn it to boolean as we have it as 1 and 0 in the csv but we want it as True and False for easier analysis
        df_merged_both_test.rename(columns={'Match': 'match'}, inplace=True)
        df_merged_both_test["match"] = df_merged_both_test["match"].astype(bool)

        #create csv that goes through all categories that are in left and right and per category get percentage of wrong predictions (match == False) and total count of items in that category
        # Create a csv with columns: category, wrong_percentage, total_count
        
        for cat_pair, count in category_pair_counts.items():

            cat_df = df_merged_both_test[df_merged_both_test["category_pair"] == cat_pair]

            y_true = cat_df["Label"]
            y_pred = cat_df["Answer_binary"]

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                average="binary",
                zero_division=0
            )

            category_analysis.append({
                "category_pair": cat_pair,
                "f1": f1,
                "total_count": len(cat_df),
                "cc": cc,
                "un": un
            })
    #create datafram for grouped cc, un, size
    category_analysis_df = pd.DataFrame(category_analysis)
    os.makedirs(f"src/models/gpt/analysis/{language}/{model}", exist_ok=True)
    category_analysis_df.to_csv(f"src/models/gpt/analysis/{language}/{model}/category_analysis_{language}_{prompt}.csv", index=False)

    #create an excel file that gets the average of f1 per category for each group of cc, un, size and model
    category_summary = (
        category_analysis_df
        .groupby(["category_pair", "cc", "un"])
        .agg(
            f1_mean=("f1", "mean"),
            total_count_mean=("total_count", "mean")
        )
        .reset_index()
    )
    os.makedirs(f"src/models/gpt/analysis/{language}/{model}", exist_ok=True)
    category_summary.to_excel(f"src/models/gpt/analysis/{language}/{model}/category_summary_{language}_{prompt}.xlsx", index=False)
        


if __name__ == "__main__":
    categories("de", "gpt-5.2", "easy_prompt")
    categories("de", "gpt-5.2", "hard_prompt")
    categories("en", "gpt-5.2", "easy_prompt")
    categories("en", "gpt-5.2", "hard_prompt")