import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

def imputer(data_):

    # Find the missing values!
    df = data_.drop(['PassengerId', 'Name',
              'Ticket', 'Cabin'], axis=1)
    print("Missing values across different columns = \n{}\n".format(df.isna().any()))
    print("Number of missing values across different columns = \n{}\n".format(df.isna().sum()))

    # Let me find the categorical variables
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = list(set(df.columns) - set(num_cols))

    # Let me OneHotEncode the categorical variables
    cat_dummies = pd.get_dummies(df[categorical_cols], drop_first=True, dtype=int)

    df = df.drop(categorical_cols,axis=1)
    for e in list(cat_dummies.columns):
        df[e] = cat_dummies[e]
    # Since kNN is used, normalization must be applied. Let's use MinMaxScaler
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
    # Now, kNNimputer can be applied to fill the nan values. The hyperparameter tuning will be performed according to how correlation between the 2 filled columns change with the target column!
    target_ = df["Survived"]

    # Let me tune the hyperparameters according to correlation coefficiency
    neigh_ = []
    metr_ = []
    wei = []
    scores_ = []
    for neighbor_ in [3,4,5,6,7,8,9,10]:
        for weights in ["uniform","distance"]:
            imputer = KNNImputer(n_neighbors=neighbor_,weights=weights)
            new_d = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)

            # Because we had multiple missing columns, we can sum the total coefficiencies and take it as a unique parameter score
            corr1 = abs(pearsonr(new_d["Age"],target_)[0])

            scores_.append(corr1)
            neigh_.append(neighbor_)
            wei.append(weights)

    index_ = scores_.index(max(scores_))

    print("Optimized Neighbor Number = {}".format(neigh_[index_]))
    print("Optimized Weights = {}".format(wei[index_]))



    return



if __name__ == "__main__":
    path_ = "C:/Users/Z840/Downloads/Kaggle/knn_imputation/train_titanic.csv"
    data_ = pd.read_csv(path_,sep=",")
    imputer(data_)
