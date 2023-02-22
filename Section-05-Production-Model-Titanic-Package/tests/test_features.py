from classification_model.config.core import config
from classification_model.processing.features import ExtractFirstLetterTransformer


def test_temporal_variable_transformer(sample_input_data):
    # Given
    #transformer = TemporalVariableTransformer(
    #    variables=config.model_config.temporal_vars,  # YearRemodAdd
    #    reference_variable=config.model_config.ref_var,
    #)
    transformer = ExtractFirstLetterTransformer(
        variables=config.model_config.cabin_vars,  # cabin
    )
    assert sample_input_data["cabin"].iat[5] == "E12"

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert subject["cabin"].iat[5] == "E"
