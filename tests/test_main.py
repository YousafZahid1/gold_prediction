import unittest
from main import load_historical_gold_data, fetch_live_gold_data, merge_gold_data

class TestMainFunctions(unittest.TestCase):

    def test_load_historical_gold_data(self):
        data = load_historical_gold_data('BSE-BOM590111.csv')
        self.assertIsNotNone(data)

    def test_fetch_live_gold_data(self):
        data = fetch_live_gold_data('GC=F')
        self.assertIsNotNone(data)

    def test_merge_gold_data(self):
        historical_data = load_historical_gold_data('BSE-BOM590111.csv')
        live_data = fetch_live_gold_data('GC=F')
        merged_data = merge_gold_data(historical_data, live_data)
        self.assertIsNotNone(merged_data)

if __name__ == '__main__':
    unittest.main()
import unittest
from main import load_historical_gold_data, fetch_live_gold_data, merge_gold_data

class TestMainFunctions(unittest.TestCase):

    def test_load_historical_gold_data(self):
        data = load_historical_gold_data('BSE-BOM590111.csv')
        self.assertIsNotNone(data)

    def test_fetch_live_gold_data(self):
        data = fetch_live_gold_data('GC=F')
        self.assertIsNotNone(data)

    def test_merge_gold_data(self):
        historical_data = load_historical_gold_data('BSE-BOM590111.csv')
        live_data = fetch_live_gold_data('GC=F')
        merged_data = merge_gold_data(historical_data, live_data)
        self.assertIsNotNone(merged_data)

if __name__ == '__main__':
    unittest.main()
