import pandas as pd
import gzip
import json
import os

# IDs that were not translated
untranslated_ids = {
    179195001984, 3776237058, 1752647304, 275958407050, 146011953933, 1859050383, 173996883983, 1918821009, 4808687386, 2183963548, 28720931869, 2027890207, 196034847138, 267280128035, 4479393573, 634067449642, 2057208876, 1847994803, 2165666882, 82466780870, 138962688072, 4720091470, 91060954192, 4062216532, 3731893848, 5035586394, 3709000540, 4479070818, 2016866277, 81174926056, 1660870123, 1802221687, 4841409785, 4833740538, 4662032123
    }

# Path to the dataset containing product information
dataset_path = "data/derived"

# Collect product details for untranslated IDs
product_details = []
for folder in ["training-sets", "validation-sets", "gold-standards_adjusted", "gold-standards"]:
    folder_path = os.path.join(dataset_path, folder)
    for file in os.listdir(folder_path):
        if file.endswith(".json.gz") and not file.__contains__("multi"):
            with gzip.open(os.path.join(folder_path, file), "rt", encoding="utf-8") as infile:
                for line in infile:
                    product = json.loads(line)
                    product_id_left = product.get("id_left")
                    product_id_right = product.get("id_right")

                    if product_id_left in untranslated_ids:
                        product_details.append({
                            "Product_ID": product_id_left,
                            "Name": product.get("name_left"),
                            "Description": product.get("desc_left")
                        })

                    if product_id_right in untranslated_ids:
                        product_details.append({
                            "Product_ID": product_id_right,
                            "Name": product.get("name_right"),
                            "Description": product.get("desc_right")
                        })

# Convert the details to a DataFrame
df = pd.DataFrame(product_details)
df = df.drop_duplicates(subset=["Product_ID"])

# Export to Excel
df.to_excel("src/translate_to_english/untranslated_ids_with_details.xlsx", index=False)

print("Untranslated IDs with details have been exported to untranslated_ids_with_details.xlsx")