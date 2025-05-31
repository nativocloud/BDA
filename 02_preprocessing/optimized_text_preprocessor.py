from pyspark.sql.functions import col, lower, regexp_replace, regexp_extract, trim, when, lit, udf
from pyspark.sql.types import StringType

def preprocess_text(df, cache=True):
    """
    Optimized text preprocessing function for Spark performance with fixed acronym handling.
    Preprocesses text by extracting optional location(s) and news source,
    normalizing acronyms, converting to lowercase, removing special characters,
    and handling multiple spaces.
    
    This version fixes the issue with acronyms like "U.S." being incorrectly converted to "u s".
    
    Args:
        df: DataFrame with text and potentially title columns.
        cache (bool): Whether to cache the preprocessed DataFrame.

    Returns:
        DataFrame: DataFrame with preprocessed text (and title if applicable),
                   plus new 'location' and 'news_source' columns.
    """
    print("Starting text preprocessing...")
    
    # Create a list to track columns that need preprocessing
    columns_to_preprocess = []
    
    # Check for text and title columns upfront to minimize schema lookups
    has_text = "text" in df.columns
    has_title = "title" in df.columns
    
    # Get column types once to avoid repeated schema lookups
    if has_text:
        text_is_string = isinstance(df.schema["text"].dataType, StringType)
        if text_is_string:
            columns_to_preprocess.append("text")
    
    if has_title:
        title_is_string = isinstance(df.schema["title"].dataType, StringType)
        if title_is_string:
            columns_to_preprocess.append("title")
    
    # --- 1. Extract Optional Location(s) and News Source from 'text' column ---
    if has_text and text_is_string:
        print("• Extracting 'location' and 'news_source' from 'text' column...")
        
        # Optimize regex pattern with non-capturing groups where possible
        news_header_pattern = r"^(?:([A-Z][a-zA-Z\s\./,]*)\s*)?\(([^)]+)\)\s*-\s*(.*)"
        
        # Apply all extractions in a single transformation to minimize passes over the data
        df = df.withColumn("location", regexp_extract(col("text"), news_header_pattern, 1)) \
               .withColumn("news_source", regexp_extract(col("text"), news_header_pattern, 2)) \
               .withColumn("text_cleaned", regexp_extract(col("text"), news_header_pattern, 3))
        
        # Update text column and handle empty extractions in a single transformation
        df = df.withColumn("text", 
                          when(col("text_cleaned") != "", col("text_cleaned"))
                          .otherwise(col("text"))) \
               .withColumn("location", 
                          when(col("location") == "", lit(None))
                          .otherwise(trim(col("location")))) \
               .withColumn("news_source", 
                          when(col("news_source") == "", lit(None))
                          .otherwise(trim(col("news_source")))) \
               .drop("text_cleaned")
        
        print("• 'location' and 'news_source' columns added (if pattern found).")
    else:
        if has_text:
            print(f"• Skipping location/source extraction: 'text' column is not a string type.")
        else:
            print("• 'text' column not found, skipping location/source extraction.")
    
    # --- 2. Apply acronym normalization BEFORE any other text processing ---
    if columns_to_preprocess:
        print(f"• Applying acronym normalization to {len(columns_to_preprocess)} column(s): {', '.join(columns_to_preprocess)}")
        
        # Define a function to normalize acronyms
        def normalize_acronyms(text):
            if text is None:
                return None
                
            # Replace common acronyms with their normalized forms
            # The order is important - longer patterns first
            replacements = [
                ("U.S.A.", "USA"),
                ("U.S.", "US"),
                ("U.N.", "UN"),
                ("F.B.I.", "FBI"),
                ("C.I.A.", "CIA"),
                ("D.C.", "DC"),
                ("U.K.", "UK"),
                ("E.U.", "EU"),
                ("N.Y.", "NY"),
                ("L.A.", "LA"),
                ("N.A.T.O.", "NATO"),
                ("W.H.O.", "WHO")
            ]
            
            for pattern, replacement in replacements:
                # Use Python's replace method which is more reliable for exact string replacement
                text = text.replace(pattern, replacement)
                
            return text
        
        # Register the UDF
        normalize_acronyms_udf = udf(normalize_acronyms, StringType())
        
        # Apply the UDF to each column that needs preprocessing
        for col_name in columns_to_preprocess:
            # First normalize acronyms using the UDF
            df = df.withColumn(col_name, normalize_acronyms_udf(col(col_name)))
            print(f"  - Applied acronym normalization to '{col_name}'.")
        
        # --- 3. Now apply the rest of the text preprocessing ---
        print(f"• Applying general text preprocessing to {len(columns_to_preprocess)} column(s)")
        
        for col_name in columns_to_preprocess:
            # Apply remaining transformations in a single chain
            df = df.withColumn(
                col_name,
                # Step 3: Trim and normalize spaces
                trim(
                    regexp_replace(
                        # Step 2: Remove special characters (keep #@)
                        regexp_replace(
                            # Step 1: Convert to lowercase
                            lower(col(col_name)),
                            "[^a-z0-9\\s#@]", " "  # Remove special chars
                        ),
                        "\\s+", " "  # Normalize spaces
                    )
                )
            )
            
            print(f"  - Applied full preprocessing chain to '{col_name}'.")
    else:
        print("• No suitable text columns found for preprocessing.")
    
    # --- 4. Data Leakage Check and Removal ('subject' column) ---
    has_subject = "subject" in df.columns
    if has_subject:
        print("\nWARNING: Removing 'subject' column to prevent data leakage.")
        df = df.drop("subject")
        print("'subject' column successfully removed.")
    else:
        print("\n'subject' column not found, no data leakage prevention needed for this column.")
    
    # --- 5. Cache the preprocessed DataFrame if requested ---
    if cache:
        print("• Caching the preprocessed DataFrame for optimized downstream operations.")
        df.cache()
        # Force materialization of the cache to ensure transformations are computed
        df.count()
    else:
        print("• Caching of the preprocessed DataFrame is disabled.")
    
    print("Text preprocessing complete.")
    return df
