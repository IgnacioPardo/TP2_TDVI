## Code

```{python}

comp_data["price_diff"] = comp_data["price"] - comp_data["original_price"]
comp_data["price_diff"] = comp_data["price_diff"].apply(lambda x: abs(x)).astype(int)

comp_data["discount"] = (comp_data["original_price"] - comp_data["price"]) / comp_data["original_price"]
comp_data["discount"] = comp_data["discount"].apply(lambda x: 1 if x == np.inf else x)

comp_data["price_ratio"] = comp_data["price"] / comp_data["original_price"]
comp_data["price_ratio"] = comp_data["price_ratio"].apply(lambda x: 1 if x == np.inf else x)

comp_data["is_discount"] = comp_data["discount"].apply(lambda x: 1 if x > 0 else 0)

comp_data["title_length"] = comp_data["title"].str.len()
comp_data["title_word_count"] = comp_data["title"].str.split(" ").apply(len)

comp_data["title_length_word_count"] = comp_data["title_length"] / comp_data["title_word_count"]
comp_data["title_length_word_count"] = comp_data["title_length_word_count"].apply(lambda x: 1 if x == np.inf else x)

comp_data["domain_dominance"] = comp_data["sold_quantity"] / comp_data["qty_items_dom"]

comp_data["is_pdp_tvi"] = comp_data["is_pdp"] / comp_data["total_visits_item"]
comp_data["is_pdp_tvi"] = comp_data["is_pdp_tvi"].apply(lambda x: 1 if x == np.inf else x)

comp_data["is_pdp_tvs"] = comp_data["is_pdp"] / comp_data["total_visits_seller"]
comp_data["is_pdp_tvs"] = comp_data["is_pdp_tvs"].apply(lambda x: 1 if x == np.inf else x)

comp_data["is_pdp_tvd"] = comp_data["is_pdp"] / comp_data["total_visits_domain"]
comp_data["is_pdp_tvd"] = comp_data["is_pdp_tvd"].apply(lambda x: 1 if x == np.inf else x)
```

## A list of the new columns with a brief description:

price_diff: This column calculates the absolute price difference between the "price" and "original_price" columns.

discount: Calculates the discount percentage for each item by comparing the difference between "original_price" and "price" with "original_price."

price_ratio: Calculates the price ratio by dividing "price" by "original_price."

is_discount: A binary column indicating whether there is a discount (1) or not (0) based on the value of the "discount" column.

title_length: Stores the character length of the "title" column for each item.

title_word_count: Counts the number of words in the "title" column by splitting the text on spaces.

title_length_word_count: Calculates the ratio of "title_length" to "title_word_count," which can help identify titles with varying levels of verbosity.

domain_dominance: Computes a measure of how dominant the item is within its domain by dividing "sold_quantity" by "qty_items_dom."

is_pdp_tvi: Calculates the ratio of "is_pdp" to "total_visits_item," representing the proportion of item visits that result in a Product Detail Page (PDP) view.

is_pdp_tvs: Calculates the ratio of "is_pdp" to "total_visits_seller," representing the proportion of seller visits that result in a PDP view.

is_pdp_tvd: Calculates the ratio of "is_pdp" to "total_visits_domain," representing the proportion of domain visits that result in a PDP view.

These columns seem to be derived from various data attributes and can be useful for analyzing and categorizing items based on their pricing, discount, title length, visit patterns, and domain dominance.

## Code

```{python}

if lab_enc:
    # LabelEncoder
    comp_data["platform"] = LabelEncoder().fit_transform(comp_data["platform"]).astype(int)
    #comp_data["category_id"] = LabelEncoder().fit_transform(comp_data["category_id"]).astype(int)
    comp_data["product_id"] = LabelEncoder().fit_transform(comp_data["product_id"]).astype(int)
    comp_data["domain_id"] = LabelEncoder().fit_transform(comp_data["domain_id"]).astype(int)
    comp_data["logistic_type"] = LabelEncoder().fit_transform(comp_data["logistic_type"]).astype(int)

if ohe:
    # OHE

    cats =  comp_data["category_id"].value_counts().index
    top_cats = comp_data["category_id"].value_counts().index[:10]

    comp_data = pd.get_dummies(comp_data,
                            columns = [
                                #"platform",
                                #"logistic_type",
                                "category_id",
                                # "domain_id"
                                ],
                            sparse = True,    # Devolver una matriz rala.
                            dummy_na = False, # No agregar columna para NaNs.
                            dtype = int       # XGBoost no trabaja con 'object'; necesitamos que sean numéricos.
                        )
    
    # drop category_id columns that are not in top_cats
    for cat in cats:
        if cat not in top_cats:
            comp_data = comp_data.drop(columns=["category_id_" + str(cat)])
```

## A list of the new columns with a brief description:

platform (LabelEncoded): If the lab_enc flag is True, this column is label-encoded to convert categorical platform data into integer values.

product_id (LabelEncoded): Label-encoded version of the "product_id" column, converting it into integer values.

domain_id (LabelEncoded): Label-encoded version of the "domain_id" column, converting it into integer values.

logistic_type (LabelEncoded): Label-encoded version of the "logistic_type" column, converting it into integer values.

One-Hot Encoding (OHE) for category_id: If the ohe flag is True, one-hot encoding is applied to the "category_id" column. It creates binary columns for each unique category and uses 1 to indicate the presence of a category and 0 otherwise. The number of columns depends on the number of unique categories in the dataset.

category_id_<category> (OHE): Binary columns created as a result of one-hot encoding for each unique category in the "category_id" column. For example, if there were categories with IDs 1, 2, 3, and so on, you would have columns like category_id_1, category_id_2, category_id_3, and so forth.

The provided code seems to be preprocessing the data by encoding categorical features into a numerical format for machine learning purposes. Label encoding is used for some columns, and one-hot encoding is applied to the "category_id" column to create binary representations of category data.




## Code

```{python}

# comp_data["boosted"] = comp_data["boosted"].astype(int)
comp_data["free_shipping"] = comp_data["free_shipping"].astype(int)
comp_data["fulfillment"] = comp_data["fulfillment"].astype(int)

comp_data["imp_is_pdp"] = comp_data["is_pdp"].isna().astype(int)
comp_data["is_pdp"].fillna(0, inplace=True)
comp_data["is_pdp"] = comp_data["is_pdp"].astype(int)

comp_data["imp_user_id"] = comp_data["user_id"].isna().astype(int)
comp_data["user_id"].fillna(0, inplace=True)
comp_data["user_id"] = comp_data["user_id"].astype(int)

comp_data["listing_type_id"] = comp_data["listing_type_id"].apply(lambda x: 0 if x == "gold_special" else 1)
```

## A list of the new columns with a brief description:

free_shipping (Type Conversion): The "free_shipping" column is converted to integers. It likely represents a binary flag where 1 indicates free shipping, and 0 indicates no free shipping.

fulfillment (Type Conversion): The "fulfillment" column is converted to integers. This column likely represents a fulfillment method, and the conversion to integers may be used for encoding different fulfillment options.

imp_is_pdp (Binary Indicator): This column is created to indicate whether the "is_pdp" column had missing values (NaN). If "is_pdp" is missing, this column will be set to 1; otherwise, it will be 0. Additionally, missing values in "is_pdp" are filled with 0 and then converted to integers.

imp_user_id (Binary Indicator): Similar to imp_is_pdp, this column is created to indicate whether the "user_id" column had missing values. If "user_id" is missing, this column will be set to 1; otherwise, it will be 0. Missing values in "user_id" are filled with 0 and then converted to integers.

listing_type_id (Mapping): The "listing_type_id" column is modified by applying a lambda function. If the value is "gold_special," it is set to 0; otherwise, it is set to 1. This appears to map different listing types to binary values for further analysis or modeling.

These modifications and additions to the columns seem to be part of data preprocessing steps, including type conversions, handling missing values, and creating binary indicators for certain conditions.



## Code

```{python}

# # For tag in tags, create a new column with the tag name and a boolean value

if add_tags:
    if len(tags) == 0:
        #tags = comp_data["tags"].str.replace("[", "").str.replace("]", "").str.split(", ").apply(pd.Series).stack().value_counts()
        # more efficient
        tags = comp_data["tags"].str.replace("[", "").str.replace("]", "").str.split(", ").explode().value_counts()
        
    for tag in tags.index:
        comp_data[tag] = comp_data["tags"].str.contains(tag).astype(int)

    comp_data["tags_count"] = comp_data["tags"].str.replace("[", "").str.replace("]", "").str.split(", ").apply(len)

    comp_data = comp_data.drop("tags", axis=1)
```

## A list of the new columns with a brief description:

Columns for Tags (if add_tags is True):

For each unique tag in the "tags" column, a new column is created with the tag name. These new columns have boolean values (1 if the item contains the tag, 0 otherwise).
The names of these columns are based on the unique tags found in the dataset.
tags_count: This column stores the count of tags associated with each item. It calculates the number of tags by splitting the "tags" column and counting the elements.

Removal of "tags" Column: After creating the new tag-related columns, the original "tags" column is dropped from the DataFrame.

The code appears to extract information from the "tags" column and create binary columns for each unique tag, indicating whether an item contains that specific tag. It also calculates the number of tags for each item and retains that count in the "tags_count" column. This can be useful for analyzing item characteristics based on their tags.




## Code

```{python}

# PolynomialFeatures custom

poly_attrs = ["print_position", "offset", "discount", "price", "health", "original_price"]

for x in poly_attrs:
    comp_data[x + "2"] = comp_data[x] ** 2

for (x, y) in itertools.combinations(poly_attrs, 2):
    comp_data[x + "2 + " + y + "2"] = comp_data[x] ** 2 + comp_data[y] ** 2
```

## A list of the new columns with a brief description:


New Columns with Squared Values:

For each attribute specified in the poly_attrs list (e.g., "print_position," "offset," "discount," "price," "health," "original_price"), a new column is created by squaring the values of that attribute. The column names follow the pattern <attribute_name>2, where <attribute_name> is the name of the original attribute.
Interaction Columns with Sum of Squares:

For each pair of attributes specified in the poly_attrs list, combinations of two attributes are created (e.g., "print_position2 + offset2"). These new columns represent the sum of squares of the corresponding attributes.
The code essentially performs feature engineering by creating new columns with squared values of specified attributes and also generates interaction features by adding the squares of pairs of attributes. These new features can capture non-linear relationships between the attributes and may be useful for machine learning models that benefit from polynomial or interaction terms.




## Code

```{python}

# Word2Vec with NLTk
RETRAIN_W2C = False

comp_data["tokenized_title"] = comp_data["title"].apply(sent_tokenize)
comp_data["tokenized_title"] = comp_data["tokenized_title"].apply(lambda x: [word_tokenize(y) for y in x])
comp_data["tokenized_title"] = comp_data["tokenized_title"].apply(lambda x: [[y2 for y2 in y1 if re.compile("[A-Za-z]").search(y2[0])] for y1 in x])
comp_data["tokenized_title"] = comp_data["tokenized_title"].apply(lambda x: [[y2.lower() for y2 in y1] for y1 in x])
        

if RETRAIN_W2C:
        
    stop_words = set(stopwords.words('spanish'))

    w2v_model = Word2Vec(vector_size=300,
                                    window=3,
                                    min_count=5,
                                    negative=15,
                                    sample=0.01,
                                    workers=8,
                                    sg=1)
                                                                        
    w2v_model.build_vocab([e2 for e1 in comp_data["tokenized_title"].values for e2 in e1],
                    progress_per=10000)

    w2v_model.train([e2 for e1 in comp_data["tokenized_title"].values for e2 in e1],
                total_examples=w2v_model.corpus_count,
                epochs=30, report_delay=1)

    w2v_model.save("title_w2c.model")

else:
    w2v_model = Word2Vec.load("title_w2c.model")

# Obtención de embeddings de títulos utilizando el modelo Word2Vec
comp_data["title_embs"] = comp_data["tokenized_title"].apply(lambda x: np.mean(
    [   
        np.zeros(w2v_model.wv.vector_size) if e2 not in w2v_model.wv 
        else w2v_model.wv.get_vector(e2) if len(e2) > 0 
        else np.zeros(w2v_model.wv.vector_size) 
        for e1 in x for e2 in e1
    ],
    axis=0)
)

comp_data["tokenized_title"] = comp_data["tokenized_title"].apply(lambda x: x[0])

comp_data[["embeddings_" + str(i) for i in range(w2v_model.wv.vector_size)]] = np.array(comp_data["title_embs"].tolist())

tokenized_title: This column is created by tokenizing the "title" column. It first uses sent_tokenize to split text into sentences, then word_tokenize to split sentences into words, removes non-alphabetical characters and converts words to lowercase. The result is a list of tokenized sentences.

Word2Vec Model Training and Loading (Conditional):

If RETRAIN_W2C is True, a Word2Vec model is trained using the tokenized titles. The model is configured with specific hyperparameters like vector size, window size, minimum word count, and others.
If RETRAIN_W2C is False, a pre-trained Word2Vec model is loaded from the file "title_w2c.model." This allows you to use a pre-existing Word2Vec model without retraining.
title_embs: This column contains the embeddings (vector representations) of the tokenized titles. It is calculated by taking the mean of word vectors for each word in the tokenized titles.

Embedding Columns: A series of columns with names like "embeddings_0," "embeddings_1," and so on are created to store the individual components of the word embeddings. These columns represent the vector components of the word embeddings generated by the Word2Vec model.

The code snippet essentially performs the following tasks: tokenizes the titles, trains or loads a Word2Vec model, calculates title embeddings, and stores the embeddings in new columns for further analysis or modeling. These embeddings can capture semantic information about the titles, making them useful for various natural language processing tasks.

## Code

```{python}

import pacmap
# Reduce dimensionality of embeddings

dims = 100

pacmap_model = pacmap.PaCMAP(
    n_components = dims,
    verbose = True,
)
embs = pacmap_model.fit_transform(comp_data[["embeddings_" + str(i) for i in range(w2v_model.wv.vector_size)]].values)

comp_data[["pacmap_" + str(i) for i in range(dims)]] = embs

comp_data = comp_data.drop(columns=["embeddings_" + str(i) for i in range(w2v_model.wv.vector_size)])
comp_data = comp_data.drop(columns=["title_embs", "tokenized_title"])
```

## A list of the new columns with a brief description:

Dimensionality Reduction with PaCMAP:

A PaCMAP (Pairwise Controlled Manifold Approximation) model is instantiated with a specified number of dimensions (dims) for dimensionality reduction.
The embeddings generated from the Word2Vec model (columns like "embeddings_0," "embeddings_1," etc.) are passed as input to the PaCMAP model for dimensionality reduction.
The fit_transform method is called to perform dimensionality reduction, and the result is stored in the embs variable.
New Columns with Reduced Dimensions:

A series of new columns is created to store the reduced-dimensional embeddings obtained from the PaCMAP model. These columns are named "pacmap_0," "pacmap_1," and so on, up to the specified number of dimensions (dims).
Column Removal:

The original embedding columns ("embeddings_0," "embeddings_1," etc.), as well as the "title_embs" and "tokenized_title" columns, are dropped from the DataFrame. This is done to remove the original embeddings and intermediate results, leaving only the reduced-dimensional embeddings.
The code snippet performs dimensionality reduction on the word embeddings using the PaCMAP technique and stores the reduced-dimensional embeddings in new columns. These reduced-dimensional representations can be used for further analysis or as input features for machine learning models, reducing the computational complexity while retaining important information captured by the embeddings.

## Code

```{python}

comp_data["warranty"] = comp_data["warranty"].str.lower()
comp_data["warranty"] = comp_data["warranty"].str.replace("á", "a")
comp_data["warranty"] = comp_data["warranty"].str.replace("í", "i")

comp_data["warranty_saler"] = comp_data["warranty"].str.contains("vendedor").astype(float)
comp_data["warranty_factory"] = comp_data["warranty"].str.contains("fabrica").astype(float)
comp_data["warranty_no"] = comp_data["warranty"].str.contains("sin garantia").astype(float)
comp_data["warranty_missing"] = (~comp_data["warranty"].isna()).astype(float)
comp_data["warranty_days"] = comp_data["warranty"].str.extract("(\d+) dias").astype(float)

def warranty_duration(warranty):
    if pd.isna(warranty):
        return np.nan
    elif "sin garantia" in warranty:
        return 0
    else:
        if "dias" in warranty:
            matches = re.findall("(\d+) dias", warranty)
            if len(matches) == 0:
                return np.nan
            else:
                return int(matches[0])
        elif "meses" in warranty:
            matches = re.findall("(\d+) meses", warranty)
            if len(matches) == 0:
                return np.nan
            else:
                return int(matches[0]) * 30    
        elif "años" in warranty:
            matches = re.findall("(\d+) años", warranty)
            if len(matches) == 0:
                return np.nan
            else:
                return int(matches[0]) * 365
        else:
            return np.nan

#list(map(lambda x: warranty_duration(x), comp_data["warranty"]))

comp_data["warranty_days"] = comp_data["warranty"].apply(warranty_duration)
comp_data["warranty_days_missing"] = (~comp_data["warranty_days"].isna()).astype(float)

# Fill ["warranty_saler", "warranty_factory", "warranty_no", "warranty_days"] with -1

comp_data["warranty_saler"] = comp_data["warranty_saler"].fillna(-1)
comp_data["warranty_factory"] = comp_data["warranty_factory"].fillna(-1)
comp_data["warranty_no"] = comp_data["warranty_no"].fillna(-1)
comp_data["warranty_days"] = comp_data["warranty_days"].fillna(-1)

comp_data.drop(columns=["warranty"], inplace=True)
```

## A list of the new columns with a brief description:

warranty Processing:

The "warranty" column is preprocessed to ensure consistent formatting:
All text is converted to lowercase.
Accented characters like "á" and "í" are replaced with their non-accented counterparts.
warranty_saler:

This column is created as a boolean flag (0 or 1) indicating whether the "warranty" text contains the word "vendedor" (seller's warranty). It is of float type with 1.0 indicating the presence of "vendedor" and 0.0 otherwise.
warranty_factory:

Similar to warranty_saler, this column is created as a boolean flag indicating whether the "warranty" text contains the word "fabrica" (factory warranty).
warranty_no:

Another boolean flag is created to indicate whether the "warranty" text contains "sin garantia" (no warranty).
warranty_missing:

This column is created as a boolean flag (0 or 1) indicating whether the "warranty" text is missing or not (NaN or not NaN).
warranty_days:

This column extracts numeric values from the "warranty" text, specifically looking for patterns like "X dias" (X days), "X meses" (X months), or "X años" (X years). It calculates the duration in days based on these patterns.
warranty_days_missing:

Similar to warranty_missing, this column is created as a boolean flag indicating whether the "warranty_days" column is missing or not (NaN or not NaN).
Filling Missing Values:

Columns warranty_saler, warranty_factory, warranty_no, and warranty_days are filled with -1 for rows where the corresponding information is missing (NaN).
warranty Column Removal:

The original "warranty" column is dropped from the DataFrame after processing.
The code snippet performs extensive preprocessing of the "warranty" column, extracts warranty-related information, and creates several binary and numeric columns to represent different aspects of warranty information for each item. These columns can be used for analysis or modeling related to product warranties.


## Code

```{python}

comp_data["date"] = pd.to_datetime(comp_data["date"])
comp_data["day"] = comp_data["date"].dt.day
comp_data["month"] = comp_data["date"].dt.month
#comp_data["hour"] = comp_data["date"].dt.hour
# comp_data["year"] = comp_data["date"].dt.year
comp_data["dayofweek"] = comp_data["date"].dt.dayofweek
# comp_data["weekofyear"] = comp_data["date"].dt.isocalendar().week
#comp_data["quarter"] = comp_data["date"].dt.quarter
# comp_data["hour"] = comp_data["date"].dt.hour
# comp_data["minute"] = comp_data["date"].dt.minute

# Morning, afternoon, night
comp_data["morning"] = (comp_data["hour"] >= 6) & (comp_data["hour"] < 12)
comp_data["morning"] = comp_data["morning"].astype(float)
comp_data["afternoon"] = (comp_data["hour"] >= 12) & (comp_data["hour"] < 18)
comp_data["afternoon"] = comp_data["afternoon"].astype(float)
comp_data["night"] = (comp_data["hour"] >= 18) & (comp_data["hour"] < 24)
```
comp_data["night"] = comp_data["night"].astype(float)
## A list of the new columns with a brief description:

Date Column Processing:

The "date" column is converted to a datetime format using pd.to_datetime.
day:

This column is created to extract and store the day component (day of the month) from the "date" column.
month:

This column is created to extract and store the month component from the "date" column.
dayofweek:

This column is created to extract and store the day of the week (0 for Monday, 6 for Sunday) from the "date" column.
Time of Day Columns (morning, afternoon, night):

These columns are created based on the "hour" component of the "date" column.
morning is set to 1 if the hour is between 6 AM (inclusive) and 12 PM (exclusive).
afternoon is set to 1 if the hour is between 12 PM (inclusive) and 6 PM (exclusive).
night is set to 1 if the hour is between 6 PM (inclusive) and 12 AM (midnight, exclusive).
These columns capture various temporal aspects of the "date" column, including day, month, day of the week, and time of day (morning, afternoon, night). These can be used for time-based analysis or as features for machine learning models to capture temporal patterns in the data. Note that there is a commented-out "hour" column creation, which might be relevant if uncommented in the code.
