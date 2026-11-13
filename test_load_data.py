import unittest
from io import StringIO

from F_LoadData import load_data


class LoadDataHeaderTests(unittest.TestCase):
    def test_default_preserves_column_names_and_first_data_row(self):
        data = StringIO("wavelength,transmission\n1500,-3\n1501,-4\n")
        result = load_data(data)
        self.assertEqual(result['wavelength_nm'].tolist(), [1500, 1501])
        self.assertEqual(result['T_dB'].tolist(), [-3, -4])

    def test_skip_header_lines_for_both_separators(self):
        for separator in (',', ';'):
            with self.subTest(separator=separator):
                data = StringIO(
                    "Instrument metadata\nMeasurement information\n"
                    f"wavelength{separator}transmission\n"
                    f"1500{separator}-3\n1501{separator}-4\n"
                )
                result = load_data(data, skiprows=2, range_wl=[1501, 1501])
                self.assertEqual(result['wavelength_nm'].tolist(), [1501])
                self.assertEqual(result['T_dB'].tolist(), [-4])


if __name__ == '__main__':
    unittest.main()
