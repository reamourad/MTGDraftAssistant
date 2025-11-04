"""
Test script for the Draft Assistant API.

Usage:
    1. Start the API: python -m uvicorn app.api:app --reload
    2. Run this script: python test_api.py
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_booster():
    """Test generating a booster pack."""
    print("=== Testing /booster ===")
    response = requests.get(f"{BASE_URL}/booster?set=MH3")

    if response.status_code == 200:
        data = response.json()
        print(f"[OK] Generated {data['count']} cards for {data['set']}")
        print(f"  Cards: {', '.join(data['pack'][:5])}...")
        return data['pack']
    else:
        print(f"[ERROR] {response.status_code} - {response.text}")
        return None


def test_predict(pack):
    """Test prediction with an empty deck."""
    print("\n=== Testing /predict ===")

    payload = {
        "set": "MH3",
        "deck": [],  # Empty deck (first pick)
        "pack": pack
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        data = response.json()
        print(f"[OK] Predictions for {data['set']}:")

        # Show top 5 recommendations
        for i, pred in enumerate(data['predictions'][:5], 1):
            prob_pct = pred['probability'] * 100
            print(f"  {i}. {pred['card_name']}: {prob_pct:.1f}%")
    else:
        print(f"[ERROR] {response.status_code} - {response.text}")


def test_predict_with_deck(pack):
    """Test prediction with cards already in deck."""
    print("\n=== Testing /predict with existing deck ===")

    # Pick first 2 cards from pack as if they're in our deck
    deck = pack[:2]
    remaining_pack = pack[2:]

    payload = {
        "set": "MH3",
        "deck": deck,
        "pack": remaining_pack
    }

    print(f"  Deck: {', '.join(deck)}")
    print(f"  Pack: {', '.join(remaining_pack[:5])}...")

    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        data = response.json()
        print(f"  Top recommendation: {data['predictions'][0]['card_name']}")
    else:
        print(f"[ERROR] {response.status_code} - {response.text}")


if __name__ == "__main__":
    print("Testing MTG Draft Assistant API\n")

    # Test booster generation
    pack = test_booster()

    if pack:
        # Test prediction with empty deck
        test_predict(pack)

        # Test prediction with cards in deck
        test_predict_with_deck(pack)

    print("\n=== Tests Complete ===")
