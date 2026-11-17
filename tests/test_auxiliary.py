"""Checks for helper interfaces, saturation, and portable audio processing."""

import tempfile
from pathlib import Path
import unittest
import numpy as np
import soundfile as sf

from section_b.data_recovery import recover_censored, em_recover_asym
from section_b.mse_visualization import ensemble_mse, mse_to_db
from section_c.speech_preprocessing import load_speech
from section_c.speech_waveform_demo import generate_waveform


class AuxiliaryTests(unittest.TestCase):
    def test_recovery_preserves_interior_observations(self):
        observed = np.array([-0.4, 0.0, 0.8])
        np.testing.assert_array_equal(
            recover_censored(observed, 0.2, -0.5, 1.5, 0.1), observed
        )

    def test_recovery_extends_beyond_saturation_bounds(self):
        recovered = em_recover_asym(np.array([-0.5, 1.5]), 0.0, 1.5, 0.5, 0.1)
        self.assertLessEqual(recovered[0], -0.5)
        self.assertGreaterEqual(recovered[1], 1.5)
        self.assertTrue(np.isfinite(recovered).all())

    def test_invalid_recovery_configuration(self):
        with self.assertRaises(ValueError):
            recover_censored(1, 0, -1, 1, 0)
        with self.assertRaises(ValueError):
            recover_censored(1, 0, 1, -1, 0.1)

    def test_mse_averaging_precedes_db_conversion(self):
        errors = np.array([[1.0, 3.0], [3.0, 1.0]])
        result = mse_to_db(ensemble_mse(errors))
        np.testing.assert_allclose(result, np.full(2, 10 * np.log10(5)))

    def test_mse_rejects_negative_values(self):
        with self.assertRaises(ValueError):
            mse_to_db([1, -1])

    def test_audio_loading_normalizes_and_pads(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.wav"
            sf.write(path, [0.0, -0.25, 0.5], 16000, subtype="DOUBLE")
            speech, rate = load_speech(path, 5)
        self.assertEqual(rate, 16000)
        np.testing.assert_array_equal(speech, [0, -0.5, 1, 0, 0])

    def test_silent_input_has_zero_echo_without_noise(self):
        result = generate_waveform(np.zeros(10), noise_std=0)
        np.testing.assert_array_equal(result["clean_echo"], 0)
        np.testing.assert_array_equal(result["observation"], 0)

    def test_waveform_seed_and_saturation(self):
        speech = np.linspace(-1, 1, 100)
        first = generate_waveform(speech, threshold=0.4)
        second = generate_waveform(speech, threshold=0.4)
        np.testing.assert_array_equal(first["observation"], second["observation"])
        self.assertLessEqual(np.max(np.abs(first["observation"])), 0.4)


if __name__ == "__main__":
    unittest.main()
