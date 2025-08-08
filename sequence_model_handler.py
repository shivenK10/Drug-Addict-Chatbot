from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SequenceModelHandler:
    def __init__(self, model_name: str):
        """
        Initialization of class arguments.

        1. model_name -> str -> Hugging face repo id.
        """
        self.model_name = model_name
    
    def load_sequence_model(self):
        """
        Loads the tokenizer and sequence classification model.

        Returns:
            model: The loaded AutoModelForSequenceClassification on CPU.
            tokenizer: The corresponding AutoTokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            device_map="cpu",
        )
        
        return model, tokenizer
