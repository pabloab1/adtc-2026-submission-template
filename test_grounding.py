import unittest
from chatbot import AgriChatbot
from chunk_retriever import retrieve

class TestGrounding(unittest.TestCase):
    def setUp(self):
        self.bot = AgriChatbot(use_llm=False)

    def test_crop_isolation(self):
        """Test that asking about one crop doesn't leak another."""
        # Turn 1: Groundnut
        self.bot.ask("how to store groundnut")
        self.assertEqual(self.bot.active_crop, "groundnut")
        
        # Turn 2: Beans
        res = self.bot.ask("how to safely store beans")
        self.assertEqual(self.bot.active_crop, "beans")
        
        # Verify no groundnut content in retrieval for turn 2
        # In use_llm=False, the answer is a string starting with [Crop - Section]
        self.assertTrue(res["answer"].startswith("[Beans"))
        self.assertNotIn("groundnut", res["answer"].lower())

    def test_all_crops(self):
        """Test all 10 supported crops retrieve correctly."""
        crops = ["beans", "cassava", "cocoa", "ginger", "groundnut", "maize", "millet", "rice", "tomato", "yam"]
        for crop in crops:
            res = self.bot.ask(f"tell me about {crop}")
            self.assertEqual(self.bot.active_crop, crop, f"Failed to activate {crop}")
            self.assertIn(crop.lower(), res["answer"].lower())
            self.assertFalse(res["low_confidence"], f"Low confidence for supported crop: {crop}")

    def test_supported_tools(self):
        """Test that supported tools return non-zero confidence."""
        tools = ["tractor", "hoe", "cutlass", "wheelbarrow", "sprayer"]
        for tool in tools:
            res = self.bot.ask(f"how to use a {tool}")
            self.assertGreater(res["confidence"], 0.35, f"Low confidence for tool: {tool}")
            self.assertIn("Farm Tools", res["answer"])

    def test_unsupported_crop(self):
        """Test that unsupported crops return low confidence and refusal."""
        res = self.bot.ask("how to plant wheat")
        self.assertTrue(res["low_confidence"])
        self.assertIn("don't have specific information", res["answer"])

    def test_gibberish(self):
        """Test that gibberish returns the specific refusal message."""
        res = self.bot.ask("asdfghjkl")
        self.assertEqual(res["answer"], "I didn't get that, please retype it.\n\nTHE DATA BASE DID NOT SPECIFICALLY SAY THIS SO BE WARNED OF MISINFORMATION")

if __name__ == "__main__":
    unittest.main()
