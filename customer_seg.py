import joblib
import pandas as pd
import datetime as dt
import numpy as np
import seaborn as sns
import missingno as msno
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from yellowbrick.cluster import KElbowVisualizer

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

###################
# FUNCTIONS
###################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def cat_summary(dataframe, col_name, plot=False):

    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)

        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)
    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def grab_col_names(dataframe, cat_th=10,  car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in df.columns if
                   dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column], color="g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return

import os
os.getcwd()

df = flo_data.copy()

df.head()
df.info()

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

analysis_date = dt.datetime(2021, 6, 1)

df["recency"] = (analysis_date - df["last_order_date"]).astype("timedelta64[D]")  #how many days since last order
df["tenure"] = (df["last_order_date"]-df["first_order_date"]).astype("timedelta64[D]")

model_df = df[["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online", "recency", "tenure"]]

for col in [col for col in model_df.columns if model_df[col].dtypes != "object"]:
    fig = plt.figure(figsize=(8,6))
    g = sns.distplot(x=model_df[col], kde=False, color="orange",
                     hist_kws=dict(edgecolor="black", linewidth=2))
    g.set_title("Column: " + str(col))
    g.xaxis.set_minor_locator(AutoMinorLocator(5))
    g.yaxis.set_minor_locator(AutoMinorLocator(5))
    g.tick_params(which="both", width=2)
    g.tick_params(which="major", length=7)
    g.tick_params(which="minor", length=4)
    plt.show()

for col in [col for col in model_df.columns if model_df[col].dtypes != "object"]:
    fig = plt.figure(figsize=(8,6))
    g = sns.distplot(x=np.log1p(model_df[col]), kde=False, color="orange",
                     hist_kws=dict(edgecolor="black", linewidth=2))
    g.set_title("Column: Logged " + str(col))
    g.xaxis.set_minor_locator(AutoMinorLocator(5))
    g.yaxis.set_minor_locator(AutoMinorLocator(5))
    g.tick_params(which="both", width=2)
    g.tick_params(which="major", length=7)
    g.tick_params(which="minor", length=4)
    plt.show()

model_df["order_num_total_ever_offline"].value_counts().sort_values()

fig = plt.figure(figsize=(8, 6))
g = sns.distplot(x=np.sqrt(model_df["recency"]), kde=False, color="orange",
                 hist_kws=dict(edgecolor="black", linewidth=2))
g.set_title("Column: Sqrt " + "recency")
g.xaxis.set_minor_locator(AutoMinorLocator(5))
g.yaxis.set_minor_locator(AutoMinorLocator(5))
g.tick_params(which="both", width=2)
g.tick_params(which="major", length=7)
g.tick_params(which="minor", length=4)
plt.show()

fig = plt.figure(figsize=(8, 6))
g = sns.distplot(x=np.cbrt(model_df["tenure"]), kde=False, color="orange",
                 hist_kws=dict(edgecolor="black", linewidth=2))
g.set_title("Column: Cbrt " + "tenure")
g.xaxis.set_minor_locator(AutoMinorLocator(5))
g.yaxis.set_minor_locator(AutoMinorLocator(5))
g.tick_params(which="both", width=2)
g.tick_params(which="major", length=7)
g.tick_params(which="minor", length=4)
plt.show()

# tenure: Cbrt
# recency: sqrt
# customer_value_total_ever_offline: log
# customer_value_total_ever_online: log

model_df["tenure"] = np.cbrt(model_df["tenure"])
model_df["recency"] = np.sqrt(model_df["recency"])
model_df["customer_value_total_ever_online"] = np.log1p(model_df["customer_value_total_ever_online"])
model_df["customer_value_total_ever_offline"] = np.log1p(model_df["customer_value_total_ever_offline"])

for col in [col for col in model_df.columns if model_df[col].dtypes != "object"]:
    fig = plt.figure(figsize=(8,6))
    g = sns.distplot(x=model_df[col], kde=False, color="orange",
                     hist_kws=dict(edgecolor="black", linewidth=2))
    g.set_title("Column: " + str(col))
    g.xaxis.set_minor_locator(AutoMinorLocator(5))
    g.yaxis.set_minor_locator(AutoMinorLocator(5))
    g.tick_params(which="both", width=2)
    g.tick_params(which="major", length=7)
    g.tick_params(which="minor", length=4)
    plt.show()

scaler = MinMaxScaler((0, 1))

scaled_df = scaler.fit_transform(model_df)

scaled_model_df = pd.DataFrame(scaled_df, columns=model_df.columns)

scaled_model_df.head()

for col in [col for col in scaled_model_df.columns if scaled_model_df[col].dtypes != "object"]:
    fig = plt.figure(figsize=(8,6))
    g = sns.distplot(x=scaled_model_df[col], kde=False, color="orange",
                     hist_kws=dict(edgecolor="black", linewidth=2))
    g.set_title("Column: " + str(col))
    g.xaxis.set_minor_locator(AutoMinorLocator(5))
    g.yaxis.set_minor_locator(AutoMinorLocator(5))
    g.tick_params(which="both", width=2)
    g.tick_params(which="major", length=7)
    g.tick_params(which="minor", length=4)
    plt.show()


from sklearn.cluster import KMeans
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(scaled_model_df)
elbow.show()

k_means = KMeans(n_clusters=7, random_state=26).fit(scaled_model_df)
segments = k_means.labels_

pd.DataFrame(segments).value_counts()

final_df = df[["master_id","order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]

final_df["segment"] = segments

final_df.groupby("segment").agg({"order_num_total_ever_online":["median","min","max"],
                                  "order_num_total_ever_offline":["median","min","max"],
                                  "customer_value_total_ever_offline":["median","min","max"],
                                  "customer_value_total_ever_online":["median","min","max"],
                                  "recency":["median", "min", "max"],
                                  "tenure":["median", "min", "max"]})