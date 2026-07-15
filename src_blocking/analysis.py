import pandas as pd
import glob
import re
import os

def map_categories(df_merged, df_categories, k, size, group_name):
    category_analysis = []
    # Merge for id_left -> shop_cat_left 
    df_merged_left = pd.merge(df_merged, df_categories[['id', 'top_category_mapped']], left_on='id_left', right_on='id', how='left', suffixes=('', '_left') ) 
    
    # Rename shop_cat column correctly 
    df_merged_left.rename(columns={'top_category_mapped': 'shop_cat_left'}, inplace=True) 
    
    # Merge for id_right -> shop_cat_right 
    df_merged_both_test = pd.merge( df_merged_left, df_categories[['id', 'top_category_mapped']], left_on='id_right', right_on='id', how='left', suffixes=('', '_right') ) 
    
    # Rename shop_cat column from right merge 
    df_merged_both_test.rename(columns={'top_category_mapped': 'shop_cat_right'}, inplace=True) 

    #print id_left and id_right and ltableid and rtableid and shop_cat_left and shop_cat_right for firrst 10 rows with nan shop_cat_left or shop_cat_right
    print(
        df_merged_both_test[
            df_merged_both_test["shop_cat_left"].isna() 
            | df_merged_both_test["shop_cat_right"].isna()
        ][["id_left", "id_right", "ltable_id", "rtable_id", "shop_cat_left", "shop_cat_right"]]
    )
    
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
    
    #make match only if label is 1 and ltableid and rtableid are not nan as that means these were not found in the blocking pairs and thus are not matched
    df_merged_both_test["match"] = (
        df_merged_both_test["ltableid"].notna()
    )

    #create csv that goes through all categories that are in left and right and per category get percentage of wrong predictions (match == False) and total count of items in that category
    # Create a csv with columns: category, wrong_percentage, total_count
    
    for cat_pair, count in category_pair_counts.items():
        cat_mask = df_merged_both_test["category_pair"] == cat_pair

        total_in_cat = cat_mask.sum()
        wrong_in_cat = ((df_merged_both_test["match"] == False) & cat_mask).sum()

        wrong_percentage = (
            wrong_in_cat / total_in_cat if total_in_cat > 0 else 0
        )

        category_analysis.append({
            "category_pair": cat_pair,
            "wrong_percentage": 1-wrong_percentage,
            "total_count": total_in_cat,
            "k": k,
            "size": size,
        })
    #create datafram for grouped cc, un, size
    category_analysis_df = pd.DataFrame(category_analysis)
    os.makedirs("data/blocking_benchmark_final/analysis/categories", exist_ok=True)
    category_analysis_df.to_csv(f"data/blocking_benchmark_final/analysis/categories/category_analysis_{group_name}_{k}_{size}.csv", index=False)

    #create an excel file that gets the average of wrong_percentage per category for each group of cc, un, size and model
    category_summary = (
        category_analysis_df
        .groupby(["category_pair", "k", "size"])
        .agg(
            wrong_percentage_mean=("wrong_percentage", "mean"),
            total_count_mean=("total_count", "mean")
        )
        .reset_index()
    )
    category_summary.to_excel(f"data/blocking_benchmark_final/analysis/categories/category_summary_{group_name}_{k}_{size}.xlsx", index=False)

def categories():
    CATEGORY_PATH = 'data/working/dedup_preprocessed_rev2_docs_since_2020_01_01_only_de_strict_only_long_name_only_mainentity_with_new_category.pkl.gz'
    tabelA_path = "data/blocking_benchmark_final/large/tableA.csv"
    tableB_path = "data/blocking_benchmark_final/large/tableB.csv"
    test_full_path = "data/blocking_benchmark_final/large/test.csv"
    predictions_path = "data/blocking_benchmark_final/results/large/blocking_pairs/*"
    df_categories = pd.read_pickle(CATEGORY_PATH, compression='gzip')
    df_tableA = pd.read_csv(tabelA_path)
    df_tableB = pd.read_csv(tableB_path)
    df_tableA["original_id"] = df_tableA["original_id"].astype(int)
    df_tableB["original_id"] = df_tableB["original_id"].astype(int)
    
    for file in glob.glob(predictions_path):
        df_test_blocking_pairs = pd.read_json(file, lines=True)
        df_test_blocking_pairs["ltableid"] = df_test_blocking_pairs["ltableid"].astype(int)
        df_test_blocking_pairs["rtableid"] = df_test_blocking_pairs["rtableid"].astype(int)
        

        name = file.split('/')[-1]
        m = re.search(
            r"blocking_pairs_(large|medium|small)_k(\d+)_test",
            name
        )

        if not m:
            raise ValueError(f"Cannot parse: {name}")

        size = m.group(1)
        k = m.group(2)

        #First get all rows from test_full that are not in the blocking pairs with label 1
        df_test_full = pd.read_csv(test_full_path)
        df_test_full["ltable_id"] = df_test_full["ltable_id"].astype(int)
        df_test_full["rtable_id"] = df_test_full["rtable_id"].astype(int)
        # drop all labels that ar enot 1
        df_test_full = df_test_full[df_test_full["label"] == 1]
    
        df_merged = pd.merge(
            df_test_full[["ltable_id", "rtable_id", "label"]],
            df_test_blocking_pairs,
            left_on=["ltable_id", "rtable_id"],
            right_on=["ltableid", "rtableid"],
            how="left"
        )

        #add to df_test_blocking pairs the original_id from table a and table b
        df_merged = pd.merge(df_merged, df_tableA[["id", "original_id"]], left_on="ltable_id", right_on="id", how="left")
        df_merged.rename(columns={"original_id": "id_left"}, inplace=True)
        df_merged = pd.merge(df_merged, df_tableB[["id", "original_id"]], left_on="rtable_id", right_on="id", how="left")
        df_merged.rename(columns={"original_id": "id_right"}, inplace=True)
        df_merged.drop(columns=["id_x", "id_y"], inplace=True)


        print(f"Mapping categories for Blocking with k={k} and size={size}")
        print("Columns in df_merged:", df_merged.columns)
        map_categories(df_merged, df_categories, k, size, "found_pairs")


if __name__ == "__main__":
    categories()        

    all_files = glob.glob("data/blocking_benchmark_final/analysis/categories/category_analysis_*.csv")

    df_list = []
    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)

    df_combined = pd.concat(df_list, ignore_index=True)

    category_summary = (
        df_combined
        .groupby(["category_pair", "k", "size"], as_index=False)
        .agg(
            wrong_percentage_mean=("wrong_percentage", "mean"),
            total_count_mean=("total_count", "mean")
        )
    )

    category_summary.to_excel(
        "data/blocking_benchmark_final/analysis/categories/category_summary_combined.xlsx",
        index=False
    )