
"""
This file contains the following:
    1. calculate_abundance
    2. calculate_completeness
    3. num_columns_break_abundance (high cardinal categorical columns)
    4. num_columns_high_vif        (columns that have very high VIF)
    5. high_missing_value_ratio    (columns that have a high missing value ratio)
"""
import pandas as pd
from warnings import filterwarnings
filterwarnings(action="ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def calculate_abundance(data):
    """
    This function calculates whether there is enough data for training model or not.
    # abundance>50 is ideal to work with
    # 25<abundance<50 is workable
    # abundance<25 is not sufficient

    THIS FUNCTION DOES NOT MUTATE THE DATA

    params:
    -------
    data(pandas.DataFrame): tabular data for which the abundance metric has to be calculated.

    returns:
    --------
    abundance(int): returns a value of abundance.
    remark(string): returns a remark associated to the value calculated
    
    """
    rows = len(data.index)
    columns = len(data.columns)
    abundance = rows / columns

    if abundance < 25:
        remark = "Data Insufficient, models will have high uncertainity"
    elif abundace >= 25 and abundance <= 50:
        remark = "Data is Just Enough, predictive models will have respectable explainability"
    else:
        "Data is Plenty, this would aid in high explainability of predictive models"

    return abundance, remark


def calculate_completeness(data):
    """
    This function calculates completeness of the data. This is done by calculating average available data points across rows and also calculates the standard deviation with a distribution plot

    THIS FUNCTION DOES NOT MUTATE THE DATA

    params:
    -------
    data(pandas.DataFrame): Data for which the completeness has to be calculated

    returns:
    --------
    completeness(float):score between 0-1
    uncertainty(float):standard deviation from completeness
    normalized_distribution:(pandas.Series): normalised distribution of missing values across rows
    
    """
    rows = len(data.index)
    columns = len(data.columns)
    distribution = data.isnull().sum(axis=1)
    normalized_distribution = distribution / columns
    completeness = normalized_distribution.mean()
    uncertainty = normalized_distribution.std()
    return completeness, uncertainty, normalized_distribution


def num_columns_break_abundance(data):
    """
    This functions calcualtes the number of categorical columns that are likely to break the abundance of the data.
    If dummification of a column reduces the the abundance metric below 25, then it is counted.

    THIS FUNCTION DOES NOT MUTATE THE DATA

    params:
    -------
    data(pandas.DataFrame): the data in which the check needs to be performed.

    returns:
    --------
    num_bad_columns(int): the number of columns in data which will break the abundance metric
    high_cardinal_columns(pandas.DataFrame): High cardinal columns in descending order of cardinality
    
    """
    import pandas as pd
    rows = len(data.index)
    columns = len(data.columns)
    categorical = data.select_dtypes(include=["object"])

    num_bad_columns = 0
    high_cardinal_columns = []
    distinct_values = []
    for column in categorical.columns:
        distinct = categorical[column].nunique(dropna=False)
        new_abundance = rows / (distinct + columns - 1)

        if new_abundance < 25:
            num_bad_columns += 1
            high_cardinal_columns.append(column)
            distinct_values.append(distinct)
    
    
    high_cardinal_df = pd.DataFrame({"column_name": high_cardinal_columns,
                                     "cardinality": distinct_values})
    high_cardinal_df.sort_values(by="cardinality", ascending=False, inplace=True)
    return num_bad_columns, high_cardinal_df


def num_columns_high_vif(data):
    """
    Detects the number of numerical columns with high Variance Inflation Factor
    VIF inhibits model interpretation and adds noise to the data

    THIS FUNCTION DOES NOT MUTATE DATA

    params:
    -------
    data(pandas.DataFrame): Data for which the high VIF columns need to be calculated

    returns:
    --------
    num_bad_columns(int): Number of columns with high VIF
    vif_df(pandas.DataFrame): list of column names with high VIF in decreasing order. Calculated iteratively.
    
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = data.select_dtypes(include=["int", "float"]).copy()
    X = X.fillna(1)

    iterative_vif_columns = []
    vif_values = []
    signal = True
    num_bad_column = 0
    while signal:
        VIF = pd.Series(
            [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
            index=X.columns,
        )
        if VIF.max() > 5:
            num_bad_column += 1
            maxi_vif = VIF.max()
            drop_column = VIF[VIF == maxi_vif].index[0]
            X.drop(columns=[drop_column], inplace=True)
            iterative_vif_columns.append(drop_column)
            vif_values.append(maxi_vif)
            # print(drop_column)
        else:
            signal = False
            
    vif_df = pd.DataFrame({"High VIF Columns":iterative_vif_columns,
                           "VIF_value": vif_values}).sort_values(by="VIF_value", ascending=False)
    return num_bad_column, vif_df


def high_missing_value_ratio(data):
    """
    This function highlights the columns which have a very high missing value ratio.
    Columns with more than 40% null values are highlighted

    THIS FUNCTION DOES NOT MUTATE DATA

    params:
    -------
    data(pandas.DataFrame)

    returns:
    --------
    num_bad_columns(int): number of columns having more than 40% null values
    bad_columns(list): Pandas Series with columns names with null value percentage in descending order
    
    """
    null_values = data.isnull().sum() / len(data) * 100
    bad_columns = null_values[null_values > 40]
    bad_columns.sort_values(ascending=False, inplace=True)
    num_bad_columns = len(bad_columns)
    mvr_df = pd.DataFrame({"Column_Name":bad_columns.index,
                           "Missing_Value_Ratio": bad_columns})
    return num_bad_columns, mvr_df


def evaluate_data(path):
    """
    This function evaluates the data using the following criteria:

    1. Abundance
    2. Completeness
    3. High Cardinal Columns
    4. High VIF
    5. Missing Value Ratio

    This function  prints the table and also saves essential results in the working directory "data metrics"

    THIS FUNCTION DOES NOT MUTATE DATA

    params:
    -------
    path(string): complete address of the data. Should be a csv file

    returns:
    --------
    metric_table(pandas.DataFrame): A table containing data quality metrics    
    
    """
    import pandas as pd
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import seaborn as sns
    import matplotlib.pyplot as plt
    from os import mkdir
    
    
    SAVE_PATH = "Data_Metrics"
    mkdir(SAVE_PATH)
    
    data = pd.read_csv(path)
    metric_table = pd.DataFrame()
    metric_table["Metric"] = []
    metric_table["Value"] = []
    metric_table["Remark"] = []

    # Calculate Abundance
    abundance, remark = calculate_abundance(data.copy())
    metric_table.loc[len(metric_table)] = ["Abundance", abundance, remark]

    # Calculate and store Completeness
    completeness, _, normalised_distribution = calculate_completeness(data.copy())
    COMPLETENESS_REMARK = "This metric indicates the average completeness of a row in dataset"
    metric_table.loc[len(metric_table)] = [
        "Completeness",
        completeness,
        COMPLETENESS_REMARK,
    ]
    plt.figure(figsize=(8, 4.5), dpi=100)
    sns.kdeplot(normalised_distribution)
    plt.title("Distribution of Completeness")
    plt.xlabel("Completeness ratio")
    plt.xlim(0,1)
    plt.savefig(fname=SAVE_PATH + "/completeness_distribution.jpg", dpi=100, format="jpg")

    # Calculate High Cardinal Columns
    high_cardinal, cardinal_df = num_columns_break_abundance(data.copy())
    CARDINAL_REMARK = "These {} columns will break the abundance of data during pre-processing".format(
        high_cardinal
    )
    cardinal_df.to_csv(SAVE_PATH + "/high_cardinal_columns.csv", index=False)
    metric_table.loc[len(metric_table)] = [
        "High Cardinal Columns",
        high_cardinal,
        CARDINAL_REMARK,
    ]

    # Calculate high VIF columns
    num_vif, vif_df = num_columns_high_vif(data.copy())
    VIF_REMARK = (
        "These {} columns are redundant and will break predictive model interpretability".format(num_vif)
    )
    vif_df.to_csv(SAVE_PATH + "/VIF_columns.csv", index=False)
    metric_table.loc[len(metric_table)] = ["High VIF Columns", num_vif, VIF_REMARK]

    # Calculating high missing value Ratio
    num_high_mvr, mvr_df = high_missing_value_ratio(data.copy())
    MVR_REMARK = "These {} columns have missing value percentage higher than 40%(acceptable threshold)".format(
        num_high_mvr
    )
    mvr_df.to_csv(SAVE_PATH + "/Missing_Value_Columns.csv", index=False)
    metric_table.loc[len(metric_table)] = [
        "High Missing Value Columns",
        num_high_mvr,
        MVR_REMARK,
    ]

    # Exporting and Printing Metric Table
    metric_table.to_csv(SAVE_PATH + "/Metric_Table.csv", index=False)
    return metric_table
