# CLAIMARC Pipeline Funnel Audit

## product_index
- `clips`: `732`
- `products`: `570`
- `products_marked_negative`: `226`
- `products_zero_comment`: `164`
- `products_by_category`: `{'apparel_and_underwear': 141, 'baby_kids_and_pets': 73, 'beauty_and_personal_care': 28, 'digital_and_electronics': 32, 'food_and_beverages': 40, 'general': 79, 'jewelry_and_collectibles': 22, 'shoes_and_bags': 73, 'smart_home': 54, 'sports_and_outdoor': 28}`

## stage_a
- `aspect_mentions`: `167398`
- `products_with_aspects`: `550`
- `unique_product_attribute_pairs`: `17485`
- `polarity`: `{'neg': 32918, 'neu': 8739, 'pos': 125741}`
- `type`: `{'attribute': 149438, 'service': 17960}`
- `explicit_fact_hit`: `{False: 157356, True: 10042}`
- `mention_strength`: `{'strong': 156041, 'weak': 11357}`
- `top_attributes`: `[('FOOD_风味', 10634), ('BEAUTY_功效', 8032), ('BEAUTY_质地', 6320), ('BEAUTY_使用方法', 2664), ('BEAUTY_适合肤质', 2557), ('FOOD_包装类型', 2300), ('BEAUTY_香味', 1886), ('GEN_价格', 1824), ('FOOD_食用方式', 1818), ('BEAUTY_包装类型', 1589), ('GEN_包装方式', 1579), ('GEN_面料材质', 1506), ('APPAREL_面料材质', 1496), ('BEAUTY_产品接收速度', 1461), ('GEN_厚度', 1419), ('BEAUTY_适用对象', 1414), ('BABY_功效', 1363), ('BABY_厚度', 1357), ('GEN_性价比', 1341), ('BABY_尺码', 1282), ('GEN_到货速度', 1264), ('BABY_适用人群', 1231), ('FOOD_产品名称', 1182), ('FOOD_到货时间', 1169), ('BABY_到货速度', 1163), ('BABY_价格', 1151), ('BABY_面料材质', 1130), ('BABY_包装方式', 1112), ('FOOD_套餐份量', 1103), ('BABY_种类', 1047)]`

## stage_b_claims
- `claim_files`: `550`
- `claim_files_nonempty`: `482`
- `claim_files_empty`: `68`
- `atomic_claims`: `11696`
- `products_with_claims`: `482`
- `products_with_claims_from_rows`: `482`
- `top_claim_attributes`: `[('APPAREL_款式', 379), ('APPAREL_颜色', 374), ('APPAREL_尺码', 246), ('DIGITAL_功能', 245), ('APPAREL_适用对象', 238), ('APPAREL_面料材质', 209), ('APPAREL_产品名称', 194), ('JEWEL_价格', 194), ('DIGITAL_售后服务', 175), ('APPAREL_价格', 164), ('DIGITAL_尺寸', 156), ('APPAREL_穿搭方式', 138), ('APPAREL_颜色分类', 134), ('BEAUTY_功效', 131), ('APPAREL_服装版型', 130), ('BABY_价格', 126), ('APPAREL_发货时效', 120), ('APPAREL_工艺', 106), ('APPAREL_厚度', 105), ('JEWEL_款式', 105), ('JEWEL_圈号', 102), ('DIGITAL_价格', 100), ('GEN_价格', 99), ('BABY_尺码', 97), ('JEWEL_品相', 96), ('APPAREL_帽子深度', 91), ('DIGITAL_使用便利性', 91), ('SHOEBAG_价格', 88), ('BEAUTY_功能', 84), ('SPORT_尺码', 79)]`

## stage_b_pairs
- `pairs`: `16679`
- `claimful_pairs`: `3514`
- `no_claim_pairs`: `13165`
- `pairs_with_aligned_negative_review`: `643`
- `category`: `{'apparel_and_underwear': 2921, 'baby_kids_and_pets': 2796, 'beauty_and_personal_care': 1500, 'digital_and_electronics': 908, 'food_and_beverages': 1673, 'general': 2904, 'jewelry_and_collectibles': 218, 'shoes_and_bags': 1540, 'smart_home': 1338, 'sports_and_outdoor': 881}`
- `claimful_by_category`: `{'apparel_and_underwear': 954, 'baby_kids_and_pets': 478, 'beauty_and_personal_care': 243, 'digital_and_electronics': 230, 'food_and_beverages': 158, 'general': 400, 'jewelry_and_collectibles': 133, 'shoes_and_bags': 502, 'smart_home': 209, 'sports_and_outdoor': 207}`
- `stats_N_total`: `{'zero': 0, 'one': 8290, 'ge2': 8389}`

## final_dataset
- `records`: `16679`
- `labels`: `{0: 16036, 1: 643}`
- `claimful`: `3514`
- `sourceful`: `7383`
- `claimful_sourceful`: `2045`
- `confidence`: `{'absent': 9296, 'medium': 2174, 'low': 5060, 'high': 149}`
- `claimful_labels`: `{0: 2871, 1: 643}`
- `claimful_sourceful_labels`: `{1: 376, 0: 1669}`
- `split`: `{'train': 11675, 'test': 3336, 'val': 1668}`
