from app.CardEncoder import CardEncoder
from app.CandidateTower import CandidateTower
import numpy as np
import numpy.testing as npt
import torch

card =   {
    "name": "Astelli Reclaimer",
    "uuid": "f6e47d02-efd5-5f85-879a-0da015bdf5b6",
    "rarity": "uncommon",
    "colors": [
      "W"
    ],
    "mana_cost": "{3}{W}{W}",
    "converted_mana_cost": 5.0,
    "types": [
      "Creature"
    ],
    "subtypes": [
      "Angel",
      "Warrior"
    ],
    "power": 5.0,
    "toughness": 4.0,
    "can_attack": True,
    "keywords": [
      "Flying",
      "Warp"
    ],
    "oracle_text": "Flying\nWhen this creature enters, return target noncreature, nonland permanent card with mana value X or less from your graveyard to the battlefield, where X is the amount of mana spent to cast this creature.\nWarp {2}{W}"
  }

card_encoder = CardEncoder(card_list=card)

#TESTING ENCODING RARIRY
npt.assert_array_equal(
    card_encoder.encode_rarity(card["rarity"]),
    np.array([0, 1, 0, 0], dtype=np.float32)
)

#TESTING ENCODING MANA COST
npt.assert_array_equal(
    card_encoder.encode_mana_cost(card["converted_mana_cost"], card["mana_cost"]),
    np.array([0.25, 0, 0, 0, 0, 0, 0.3125], dtype=np.float32)
)

#TESTING ENCODING TYPES
npt.assert_array_equal(
    card_encoder.encode_types(card["types"]),
    np.array([0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
)

#TESTING ENCODING POWER AND TOUGHNESS
npt.assert_array_equal(
    card_encoder.encode_power_toughhness(card["can_attack"], card["power"], card["toughness"]),
    np.array([1, 5/15, 4/15], dtype=np.float32)
)

candidateTower = CandidateTower()
encoded = card_encoder.encode(card)
card_tensor = torch.from_numpy(encoded).unsqueeze(0)  # (1, 424)
x = candidateTower.forward(card_tensor)

print(x)
print(len(x))
print('Test passed successfully.')