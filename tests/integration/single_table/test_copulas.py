from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from sdv.datasets.demo import download_demo
from sdv.single_table import GaussianCopulaSynthesizer


def test_adding_constraints():
    """End to end test for the ``SDV (Advanced): Adding Constraints.ipynb``."""
    # Setup
    real_data, metadata = download_demo(
        modality='single_table',
        dataset_name='fake_hotel_guests'
    )

    checkin_lessthan_checkout = {
        'constraint_class': 'Inequality',
        'constraint_parameters': {
            'low_column_name': 'checkin_date',
            'high_column_name': 'checkout_date'
        }
    }
    synthesizer = GaussianCopulaSynthesizer(metadata)

    # Run
    synthesizer.add_constraints([checkin_lessthan_checkout])
    synthesizer.fit(real_data)

    synthetic_data_constrained = synthesizer.sample(500)

    synthetic_dates = synthetic_data_constrained[['checkin_date', 'checkout_date']].dropna()
    checkin_dates = pd.to_datetime(synthetic_dates['checkin_date'])
    checkout_dates = pd.to_datetime(synthetic_dates['checkout_date'])
    violations = checkin_dates >= checkout_dates

    assert all(~violations)

    # Load custom constraint class
    synthesizer.load_custom_constraint_classes(
        'tests/integration/single_table/custom_constraints.py',
        ['IfTrueThenZero']
    )
    rewards_member_no_fee = {
        'constraint_class': 'IfTrueThenZero',
        'constraint_parameters': {
            'column_names': ['has_rewards', 'amenities_fee']
        }
    }

    synthesizer.add_constraints([
        rewards_member_no_fee
    ])

    # Re-Fit the model
    synthesizer.fit(real_data)
    synthetic_data_custom_constraint = synthesizer.sample(500)

    # Assert
    validation = synthetic_data_custom_constraint[synthetic_data_custom_constraint['has_rewards']]
    assert validation['amenities_fee'].sum() == 0.0

    # Save and Load
    temp_dir = TemporaryDirectory()
    model_path = Path(temp_dir.name) / 'synthesizer.pkl'
    synthesizer.save(model_path)

    # Assert
    assert model_path.exists()
    assert model_path.is_file()
    loaded_synthesizer = GaussianCopulaSynthesizer.load(model_path)

    assert isinstance(loaded_synthesizer, GaussianCopulaSynthesizer)
    assert loaded_synthesizer.get_info() == synthesizer.get_info()
    assert loaded_synthesizer.metadata.to_dict() == metadata.to_dict()
    sampled_data = loaded_synthesizer.sample(100)
    validation = sampled_data[sampled_data['has_rewards']]
    assert validation['amenities_fee'].sum() == 0.0
