import pandas as pd
import torch
import unittest

from boolq import BoolQDataset
from transformers import RobertaTokenizerFast


class TestBoolQDataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.dataset = pd.DataFrame.from_dict(
            {
                "question": ["question 0", "question 1"],
                "passage": ["passage 0", "passage 1"],
                "idx": [0, 1],
                "label": [True, False],
            }
        )
        self.max_seq_len = 4
        self.boolq_dataset = BoolQDataset(
            self.dataset, self.tokenizer, self.max_seq_len
        )

    def test_len(self):
        ## Test that the length of self.boolq_dataset is correct.
        ## len(self.boolq_dataset) should equal len(self.dataset).
        self.assertEqual(self.boolq_dataset.__len__(), len(self.dataset))

    def test_item(self):
        ## Test that, for each element of self.boolq_dataset, 
        ## the output of __getitem__ (accessible via self.boolq_dataset[idx])
        ## has the correct keys, value dimensions, and value types.
        ## Each item should have keys ["input_ids", "attention_mask", "labels"].
        ## The input_ids and attention_mask values should both have length self.max_seq_len
        ## and type torch.long. The labels value should be a single numeric value.
        for idx in [0, 1]:
            item = self.boolq_dataset[idx]
            self.assertEqual(list(item.keys()), ["input_ids", "attention_mask", "labels"])
            self.assertEqual(item['input_ids'].shape[0], self.max_seq_len)
            self.assertEqual(item['attention_mask'].shape[0], self.max_seq_len)
            self.assertEqual(item['input_ids'].dtype, torch.long)
            self.assertEqual(item['attention_mask'].dtype, torch.long)
            self.assertEqual(type(item['labels']), int)



if __name__ == "__main__":
    unittest.main()
