#model.py

from transformers import Qwen2Model, Qwen2PreTrainedModel, AutoTokenizer , BitsAndBytesConfig
import torch
import torch.nn as nn

TAXONOMY_TO_INDEX = {
    "adaptive": {
        "A":   {1: 0, 3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6},
        "B-O": {1: 7, 3: 8},
        "B-S": {1: 9},
        "C-O": {1: 10, 3: 11},
        "C-S": {1: 12},
        "D":   {1: 13, 3: 14, 5: 15},
    },
    "maladaptive": {
        "A":   {2: 16, 4: 17, 6: 18, 8: 19, 10: 20, 12: 21, 14: 22},
        "B-O": {2: 23, 4: 24},
        "B-S": {2: 25},
        "C-O": {2: 26, 4: 27},
        "C-S": {2: 28},
        "D":   {2: 29, 4: 30, 6: 31},
    }
}

INDEX_TO_TAXONOMY = {}
for valence, elements in TAXONOMY_TO_INDEX.items():
    for element, subelements in elements.items():
        for number, index in subelements.items():
            INDEX_TO_TAXONOMY[index] = {"valence": valence, "element": element, "number": number}

ELEMENT_SLICES = {
    "adaptive": {
        "A": (0, 7), "B-O": (7, 9), "B-S": (9, 10), 
        "C-O": (10, 12), "C-S": (12, 13), "D": (13, 16)
    },
    "maladaptive": {
        "A": (16, 23), "B-O": (23, 25), "B-S": (25, 26), 
        "C-O": (26, 28), "C-S": (28, 29), "D": (29, 32)
    }
}



class QwenSelfStatePredictor(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.num_subelements = 32
        self.num_presence = 2
        self.classifier = nn.Linear(config.hidden_size, self.num_subelements + self.num_presence)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # Get hidden state of the last non-padded token
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        last_hidden_states = outputs.last_hidden_state[torch.arange(batch_size, device=input_ids.device), sequence_lengths]
        
        logits = self.classifier(last_hidden_states)
        subelement_logits = logits[:, :self.num_subelements]
        presence_preds = logits[:, self.num_subelements:]
        
        # Loss calculation removed from forward pass
        return {"subelement_logits": subelement_logits, "presence_preds": presence_preds}

def decode_predictions(subelement_logits, presence_preds, threshold=0.5):
    """Translates tensors to the JSON dictionary format expected by CLPsych."""
    probs = torch.sigmoid(subelement_logits)
    
    ada_presence = max(1, min(5, round(presence_preds[0].item())))
    mal_presence = max(1, min(5, round(presence_preds[1].item())))
    
    prediction = {
        "adaptive-state": {"Presence": ada_presence},
        "maladaptive-state": {"Presence": mal_presence}
    }
    
    for valence in ["adaptive", "maladaptive"]:
        state_key = f"{valence}-state"
        
        for element, (start_idx, end_idx) in ELEMENT_SLICES[valence].items():
            element_probs = probs[start_idx:end_idx]
            max_prob, max_local_idx = torch.max(element_probs, dim=0)
            
            if max_prob.item() >= threshold:
                global_idx = start_idx + max_local_idx.item()
                predicted_number = INDEX_TO_TAXONOMY[global_idx]["number"]
                prediction[state_key][element] = {"subelement": predicted_number}

        # Spec constraint: if no subelements, presence MUST be 1
        if len(prediction[state_key]) == 1: # Only 'Presence' key exists
            prediction[state_key]["Presence"] = 1

    return prediction