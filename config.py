import os

# Paths
DATA_DIR = '../input/competitions/house-prices-advanced-regression-techniques'
OUTPUT_DIR = '../outputs'

# File paths
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Feature engineering
NUMERIC_FEATURES = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 
                    'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
                    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 
                    '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
                    'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                    'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
                    'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                    'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
                    'MiscVal', 'MoSold', 'YrSold']

CATEGORICAL_FEATURES = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
                        'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
                        'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
                        'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
                        'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
                        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                        'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir',
                        'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
                        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                        'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
                        'SaleType', 'SaleCondition']

# Columns to fill with 'None'
NONE_COLUMNS = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                'BsmtFinType2', 'MasVnrType']

# Columns to fill with 0
ZERO_COLUMNS = ['GarageYrBlt', 'MasVnrArea']