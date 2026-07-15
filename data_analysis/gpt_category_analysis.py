import pandas as pd
from sklearn.metrics import f1_score
import os
import gzip
import json

CATEGORY_PATH = 'data/working/dedup_preprocessed_rev2_docs_since_2020_01_01_only_de_strict_only_long_name_only_mainentity_with_new_category.pkl.gz'

# Process record GPT prompt
def process_record(record):

    # Extract 'left' values (excluding ID)
    left_parts = [
        "Marke: "+ str(record.get("brand_left")) if record.get("brand_left") is not None else "" ,
        "Name: "+str(record.get("name_left")) if record.get("name_left") is not None else "" ,
        "Preis: "+str(record.get("price_left")) if record.get("price_left") is not None else "",
        "Beschreibung: " + str(record.get("desc_left")) if record.get("desc_left") is not None else ""
    ]
    left_text = " ".join(filter(None, left_parts))  # filter out empty strings

    # Extract 'right' values (excluding ID)
    right_parts = [
        "Marke: "+ str(record.get("brand_right")) if record.get("brand_right") is not None else "",
        "Name: "+str(record.get("name_right")) if record.get("name_right") is not None else "",
        "Preis: "+str(record.get("price_right")) if record.get("price_right") is not None else "",
        "Beschreibung: " +str(record.get("desc_right")) if record.get("desc_right") is not None else "" 
    ]
    right_text = " ".join(filter(None, right_parts))  # filter out empty strings


    # Example of further processing (modify as required)
    left_text = left_text.replace("/", " ")  # Replace slashes with spaces
    right_text = right_text.replace("/", " ")  # Replace slashes with spaces

    return left_text, right_text

def categories_fro_json_row(record): 
    category_1 = record.get("new_category_left") 
    category_2 = record.get("new_category_right") 
    
    return category_1, category_2

def test_hard_categories_gpt(model, cc, un, batched = False):
    if batched:
        batched = "_batched"
    else:
        batched = ""
    RESULTS = f'src/models/gpt/reports/{model}/products_{cc}_{un}un{batched}.csv' 
    TEST_PATH = f'data/derived/gold-standards/products{cc}rnd{un}un_gs.json.gz' 
    # Add the shop_cat from CATEGORY_PATH top the TEST_PATH dataset by connecting the shop_cat by id_left and id_right 
    df_test = pd.read_json(TEST_PATH, lines=True, compression='gzip') 
    df_category = pd.read_pickle(CATEGORY_PATH, compression='gzip') 
    
    # Merge for id_left -> shop_cat_left 
    df_merged_left = pd.merge( df_test, df_category[['id', 'top_category_mapped']], left_on='id_left', right_on='id', how='left', suffixes=('', '_left') ) 
    
    # Rename shop_cat column correctly 
    df_merged_left.rename(columns={'top_category_mapped': 'shop_cat_left'}, inplace=True) 
    
    # Merge for id_right -> shop_cat_right 
    df_merged_both_test = pd.merge( df_merged_left, df_category[['id', 'top_category_mapped']], left_on='id_right', right_on='id', how='left', suffixes=('', '_right') ) 
    
    # Rename shop_cat column from right merge 
    df_merged_both_test.rename(columns={'top_category_mapped': 'shop_cat_right'}, inplace=True) 
    results_df = pd.read_csv(RESULTS) 
    
    non_matches_shop_cats = [] 
    with gzip.open(TEST_PATH, "rt", encoding="utf-8") as file: 
        for i, line in enumerate(file): 
            record = json.loads(line) 
            # Load each JSON object separately 
            entity_1, entity_2 = process_record(record) 
            
            # Process data 
            # check if results_df contains a record where entity_1 is in Entity1 and entity_2 is in Entity2 
            match_row = results_df[(results_df['Entity1'] == entity_1) & (results_df['Entity2'] == entity_2)] 
            if not match_row.empty:
                if match_row['Match'].iloc[0] == 0:
                    shop_cat_left = df_merged_both_test.loc[df_merged_both_test['pair_id'] == record['pair_id'], 'shop_cat_left'].values
                    shop_cat_right = df_merged_both_test.loc[df_merged_both_test['pair_id'] == record['pair_id'], 'shop_cat_right'].values
                    non_matches_shop_cats.append((shop_cat_left[0], shop_cat_right[0], entity_1, entity_2))



    # Now print amount where shop_cat_left != shop_cat_right
    different_category_count = sum(1 for cat_left, cat_right, entity_1, entity_2 in non_matches_shop_cats if cat_left != cat_right)
    print(f"Total non-matches: {len(non_matches_shop_cats)}")
    print(f"Non-matches with different shop categories: {different_category_count}")
    
    with open(f'src/models/gpt/reports/{model}/matches_of_different_categories_{cc}_{un}un.txt', 'w', encoding='utf-8') as f:
        for cat_left, cat_right, entity_1, entity_2 in non_matches_shop_cats:
            if cat_left != cat_right:
                f.write(f"Non-match categories: Left - {cat_left}, Right - {cat_right}\n")
                f.write(f"Entity 1: {entity_1} \n")
                f.write(f"Entity 2: {entity_2} \n")
                f.write("-----\n") 

    #get amount of items grouped by category from df_merged_both_test
    category_counts = (
        df_merged_both_test[['shop_cat_left', 'shop_cat_right']]
        .stack()
        .value_counts(dropna=False)
        .to_dict()
    )


    # get percentage of each category matche and non matches
    with open(f'src/models/gpt/reports/{model}/category_evaluation_{cc}_{un}un.csv', 'w', encoding='utf-8') as f:
        f.write("Category,Total,Non-matches,Percentage Non-matches\n")
        for (cat, amount) in category_counts.items():
            non_matches_in_cat = sum(1 for cat_left, cat_right,entity_1, entity_2 in non_matches_shop_cats if cat_left == cat or cat_right == cat)
            percentage_non_matches = (non_matches_in_cat / amount) * 100
            #print(f"Category: {cat}, Total: {amount}, Non-matches: {non_matches_in_cat}, Percentage Non-matches: {percentage_non_matches:.2f}%")
            f.write(f"{cat},{amount},{non_matches_in_cat},{percentage_non_matches:.2f}\n")


if __name__ == "__main__":
    #test_hard_categories_gpt('gpt_4o_mini', "80cc20", "100")
    #test_hard_categories_gpt('gpt_5_mini', "80cc20", "100")
    #test_hard_categories_gpt('gpt_5_mini', "50cc50", "100")
    test_hard_categories_gpt('gpt-5.2', "80cc20", "100")
    test_hard_categories_gpt('gpt-5.2', "80cc20", "050", True)