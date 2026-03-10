import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder


def feature_engineering(df):

    df = df.copy()

    # Cabin features
    df['Deck'] = df['Cabin'].apply(
        lambda x: x.split('/')[0] if pd.notna(x) else 'Unknown'
    )

    df['Cabin_num'] = df['Cabin'].apply(
        lambda x: int(x.split('/')[1]) if pd.notna(x) else -1
    )

    df['Side'] = df['Cabin'].apply(
        lambda x: x.split('/')[2] if pd.notna(x) else 'Unknown'
    )

    # Passenger group
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    df['Group_size'] = df.groupby('Group')['Group'].transform('count')
    df['Solo'] = (df['Group_size'] == 1).astype(int)

    # Name features
    df['FirstName'] = df['Name'].apply(
        lambda x: x.split()[0] if pd.notna(x) else 'Unknown'
    )

    df['LastName'] = df['Name'].apply(
        lambda x: x.split()[-1] if pd.notna(x) else 'Unknown'
    )

    df['Family_size'] = df.groupby('LastName')['LastName'].transform('count')

    # Spending features
    spending_cols = [
        'RoomService',
        'FoodCourt',
        'ShoppingMall',
        'Spa',
        'VRDeck'
    ]

    df['TotalSpending'] = df[spending_cols].sum(axis=1)

    df['HasSpending'] = (df['TotalSpending'] > 0).astype(int)
    df['NoSpending'] = (df['TotalSpending'] == 0).astype(int)

    for col in spending_cols:
        df[f'{col}_ratio'] = df[col] / (df['TotalSpending'] + 1)

    # Age group
    df['Age_group'] = pd.cut(
        df['Age'],
        bins=[0,12,18,30,50,100],
        labels=[
            'Child',
            'Teen',
            'Young_Adult',
            'Adult',
            'Senior'
        ]
    ).astype(str)

    # Missing indicators
    df['Age_missing'] = df['Age'].isna().astype(int)
    df['CryoSleep_missing'] = df['CryoSleep'].isna().astype(int)

    return df


def preprocess_data(df, is_train=True):

    df = df.copy()

    categorical_features = [
        'HomePlanet',
        'CryoSleep',
        'Destination',
        'VIP',
        'Deck',
        'Side',
        'Age_group'
    ]

    numerical_features = [
        'Age',
        'RoomService',
        'FoodCourt',
        'ShoppingMall',
        'Spa',
        'VRDeck',
        'Cabin_num',
        'Group_size',
        'Solo',
        'Family_size',
        'TotalSpending',
        'HasSpending',
        'NoSpending',
        'Age_missing',
        'CryoSleep_missing'
    ] + [col for col in df.columns if '_ratio' in col]


    # Fill missing categorical
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')


    # Convert numeric columns
    for col in numerical_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')


    # Fill missing numerical
    for col in numerical_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())


    # Encode categorical
    label_encoders = {}

    for col in categorical_features:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le


    # Save encoders during training
    if is_train:
        with open("model/label_encoders.pkl", "wb") as f:
            pickle.dump(label_encoders, f)


    feature_columns = categorical_features + numerical_features

    feature_columns = [
        col for col in feature_columns
        if col in df.columns
    ]

    X = df[feature_columns]


    if is_train:
        y = df['Transported'].astype(int)
        return X, y, feature_columns

    else:
        return X, feature_columns
        
