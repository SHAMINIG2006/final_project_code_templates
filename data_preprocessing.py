import pandas as pd

def preprocess_data(csv_path):
    """
    Loads and preprocesses the dataset.

    Args:
        csv_path (str): Path to the CSV file

    Returns:
        X (DataFrame): Features
        y (Series): Target variable
    """

    # Load dataset
    df = pd.read_csv('/data/PS_20174392719_1491204439457_log.csv')

    # Drop unnecessary columns
    df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)

    # Encode transaction type
    df['type'] = df['type'].map({
        'CASH_OUT': 0,
        'TRANSFER': 1,
        'PAYMENT': 2,
        'DEBIT': 3,
        'CASH_IN': 4
    })

    # Features and label
    X = df.drop('isFraud', axis=1)
    y = df['isFraud']

    return X, y
