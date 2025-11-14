"""
Fetch training data from Cloudflare Worker
Downloads all stored game states for ML training
"""
import requests
import json
import os
from datetime import datetime

# Configure your Worker URL
WORKER_URL = "https://alvin-pacman-ai.jhaladik.workers.dev"
OUTPUT_DIR = "data/raw"

def fetch_training_data():
    """Download all training data from Cloudflare Worker"""
    print("Fetching training data from Cloudflare Worker...")
    print(f"URL: {WORKER_URL}/api/export-training-data")

    try:
        response = requests.get(f"{WORKER_URL}/api/export-training-data", timeout=30)
        response.raise_for_status()

        data = response.json()

        if not data.get('success'):
            print(f"Error: {data.get('error', 'Unknown error')}")
            return None

        count = data.get('count', 0)
        training_data = data.get('data', [])
        export_date = data.get('exportDate', datetime.now().isoformat())

        print(f"✓ Successfully fetched {count} training samples")
        print(f"✓ Export date: {export_date}")

        # Save raw data
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_DIR, f"training_data_{timestamp}.json")

        with open(output_file, 'w') as f:
            json.dump({
                'count': count,
                'exportDate': export_date,
                'data': training_data
            }, f, indent=2)

        print(f"✓ Saved to: {output_file}")

        # Print statistics
        print("\nData Statistics:")
        print(f"  Total samples: {count}")

        if count > 0:
            actions = {}
            rewards_sum = 0
            for sample in training_data:
                action = sample.get('action', 'unknown')
                actions[action] = actions.get(action, 0) + 1
                rewards_sum += sample.get('reward', 0)

            print(f"  Average reward: {rewards_sum / count:.2f}")
            print(f"  Action distribution:")
            for action, cnt in sorted(actions.items()):
                print(f"    {action}: {cnt} ({cnt/count*100:.1f}%)")

        return training_data

    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching data: {e}")
        return None

def main():
    print("=" * 60)
    print("Alvin Pac-Man AI - Training Data Export")
    print("=" * 60)
    print()

    data = fetch_training_data()

    if data is None:
        print("\n✗ Failed to fetch training data")
        print("Make sure:")
        print("  1. Worker is deployed and running")
        print("  2. /api/export-training-data endpoint is accessible")
        print("  3. You have played some games to generate data")
        return 1

    if len(data) < 100:
        print(f"\n⚠ Warning: Only {len(data)} samples available")
        print("   Recommendation: Play more games to collect at least 1000 samples")
        print("   for effective training")

    print("\n✓ Data export complete!")
    print("\nNext steps:")
    print("  1. cd ml-training")
    print("  2. python preprocessing/feature_extraction.py")
    print("  3. python training/train.py")

    return 0

if __name__ == "__main__":
    exit(main())
