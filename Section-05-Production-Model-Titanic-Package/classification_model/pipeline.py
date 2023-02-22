from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder, OneHotEncoder
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,  
)
from feature_engine.selection import DropFeatures
from feature_engine.transformation import LogTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, MinMaxScaler, StandardScaler

from classification_model.config.core import config
from classification_model.processing import features as pp

price_pipe = Pipeline(
    [
        # ===== IMPUTATION =====
        # impute categorical variables with string missing
        (
            'categorical_imputation', 
            CategoricalImputer(
                imputation_method='missing', 
                variables = config.model_config.categorical_vars, 
                )
        ),
        # add missing indicator
        (
            "missing_indicator",
            AddMissingIndicator(variables=config.model_config.numerical_vars
            ),
        ),
        # impute numerical variables with the mean
        (
            "mean_imputation",
            MeanMedianImputer(
                imputation_method="mean",
                variables=config.model_config.numerical_vars,
            ),
        ),
        (
            "extract_letter",
            pp.ExtractFirstLetterTransformer(
                variables=config.model_config.cabin_vars,
            ),
        ),
        #("drop_features", DropFeatures(features_to_drop=[config.model_config.unused_vars])),
        # ==== VARIABLE TRANSFORMATION =====
        #("log", LogTransformer(variables=config.model_config.numericals_log_vars)),
        # === mappers ====
        # No Mapping Done here
        # == CATEGORICAL ENCODING
        # encode categorical variables using the target mean
         (
            "rare_label_encoder",
            RareLabelEncoder(
                tol=0.01, n_categories=1, variables=config.model_config.categorical_vars
            ),
        ),
        (
            "categorical_encoder", 
            OneHotEncoder(
                drop_last = True, variables = config.model_config.categorical_vars
            ),
        ),
        (
            "scaler", 
            StandardScaler()
        ),
        (
            "Lasso",
            LogisticRegression(
                C=config.model_config.alpha,
                random_state=config.model_config.random_state,
            ),
        ),
    ]
)
