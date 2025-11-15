"""
Export Training Data from Cloudflare Worker
Downloads all human gameplay data from KV storage via API
"""
import requests
import json
import os
from datetime import datetime


def export_training_data(worker_url, output_path='data/training_data.json'):
    """
    Export training data from Cloudflare Worker

    Args:
        worker_url: URL of deployed Cloudflare Worker
        output_path: Where to save the JSON file
    """
    print("=" * 70)
    print("Exporting Training Data from Cloudflare")
    print("=" * 70)
    print()

    # Call export endpoint
    export_url = f"{worker_url}/api/export-training-data"
    print(f"Fetching data from: {export_url}")

    try:
        response = requests.get(export_url)
        response.raise_for_status()

        data = response.json()

        if not data.get('success'):
            print(f"[ERROR] Export failed: {data.get('error', 'Unknown error')}")
            return False

        # Statistics
        count = data.get('count', 0)
        export_date = data.get('exportDate', datetime.now().isoformat())

        print(f"[+] Exported {count} samples")
        print(f"[+] Export date: {export_date}")

        if count == 0:
            print("[WARNING] No data found. Play some games first!")
            return False

        # Analyze data
        samples = data.get('data', [])
        if samples:
            # Check vector dimensions
            vector_dims = set(len(s['vector']) for s in samples if 'vector' in s)
            print(f"[+] Vector dimensions found: {vector_dims}")

            # Check actions
            actions = set(s['action'] for s in samples if 'action' in s)
            print(f"[+] Actions: {actions}")

            # Reward stats
            rewards = [s['reward'] for s in samples if 'reward' in s]
            if rewards:
                print(f"[+] Reward range: [{min(rewards):.1f}, {max(rewards):.1f}]")
                print(f"[+] Mean reward: {sum(rewards)/len(rewards):.2f}")

        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print()
        print(f"[+] Saved to: {output_path}")
        print()
        print("=" * 70)
        print("Export Complete!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("1. Train model: python train_128.py")
        print("2. Deploy to HF Spaces")
        print()

        return True

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch data: {e}")
        return False


def main():
    # Configuration
    WORKER_URL = "https://alvin-pacman-ai.jhaladik.workers.dev"
    OUTPUT_PATH = "data/training_data.json"

    # Export
    success = export_training_data(WORKER_URL, OUTPUT_PATH)

    if not success:
        print("\n[!] Export failed. Check your Worker URL and make sure you have training data.")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
