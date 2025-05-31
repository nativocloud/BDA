from pyspark.sql.functions import col, lower, regexp_replace, regexp_extract, trim, when, lit
from pyspark.sql.types import StringType

def preprocess_text(df, cache=True):
    """
    Optimized text preprocessing function for Spark performance.
    Preprocesses text by extracting optional location(s) and news source,
    normalizing acronyms, converting to lowercase, removing special characters,
    and handling multiple spaces.
    
    This version is optimized for Spark performance with:
    - Minimized DataFrame transformations
    - Chained transformations where possible
    - Strategic column pruning
    - Optimized regex patterns
    - Explicit memory management
    
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
    
    # --- 2. Apply general preprocessing steps to text columns ---
    if columns_to_preprocess:
        print(f"• Applying general preprocessing to {len(columns_to_preprocess)} column(s): {', '.join(columns_to_preprocess)}")
        
        # Create a dictionary of acronym replacements for more efficient processing
        acronyms = {
            r"\bU\.S\.\b": "US ",
            r"\bU\.N\.\b": "UN ",
            r"\bF\.B\.I\.\b": "FBI ",
            r"\bC\.I\.A\.\b": "CIA "
            # Add other acronyms here
        }
        
        # Process each column that needs preprocessing
        for col_name in columns_to_preprocess:
            # Chain multiple transformations to minimize DataFrame creation
            # Start with acronym normalization (must happen before lowercase)
            for pattern, replacement in acronyms.items():
                df = df.withColumn(col_name, regexp_replace(col(col_name), pattern, replacement))
            
            # Apply remaining transformations in a single chain
            df = df.withColumn(
                col_name,
                # Step 1: Convert to lowercase
                trim(
                    # Step 3: Replace multiple spaces with single space and trim
                    regexp_replace(
                        # Step 2: Remove special characters (keep #@)
                        regexp_replace(
                            lower(col(col_name)),  # Step 1: Lowercase
                            "[^a-z0-9\\s#@]", " "  # Step 2: Remove special chars
                        ),
                        "\\s+", " "  # Step 3: Normalize spaces
                    )
                )
            )
            
            print(f"  - Applied full preprocessing chain to '{col_name}'.")
    else:
        print("• No suitable text columns found for preprocessing.")
    
    # --- 3. Data Leakage Check and Removal ('subject' column) ---
    has_subject = "subject" in df.columns
    if has_subject:
        print("\nWARNING: Removing 'subject' column to prevent data leakage.")
        df = df.drop("subject")
        print("'subject' column successfully removed.")
    else:
        print("\n'subject' column not found, no data leakage prevention needed for this column.")
    
    # --- 4. Cache the preprocessed DataFrame if requested ---
    if cache:
        print("• Caching the preprocessed DataFrame for optimized downstream operations.")
        df.cache()
        # Force materialization of the cache to ensure transformations are computed
        df.count()
    else:
        print("• Caching of the preprocessed DataFrame is disabled.")
    
    print("Text preprocessing complete.")
    return df
