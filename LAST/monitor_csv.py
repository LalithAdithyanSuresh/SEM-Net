import argparse
import os
import time
import csv
import requests
import numpy as np

def send_to_dashboard(url, payload):
    try:
        requests.post(url, json=payload, timeout=3)
        print(f"Sent live update to dashboard: {payload}")
    except Exception as e:
        print(f"Failed to update dashboard: {e}")

def parse_csv(csv_path, total_expected):
    if not os.path.exists(csv_path):
        return 0, 0.0, False

    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Check if the last row is the AVERAGE summary
        has_average = False
        avg_row = None
        for r in rows:
            if r and r[0] == 'AVERAGE':
                has_average = True
                avg_row = r
                break

        if has_average and avg_row:
            try:
                avg_psnr = float(avg_row[1])
                return total_expected, avg_psnr, True
            except Exception:
                pass

        # Parse active rows
        psnrs = []
        for r in rows[1:]:
            if r and len(r) >= 5 and r[0].strip() and r[0] != 'Image' and r[0] != 'AVERAGE':
                try:
                    psnrs.append(float(r[1]))
                except ValueError:
                    pass
        
        count = len(psnrs)
        avg_psnr = np.mean(psnrs) if count > 0 else 0.0
        return count, avg_psnr, False

    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return 0, 0.0, False

def main():
    parser = argparse.ArgumentParser(description="Monitor evaluation CSV progress and send live updates to C2 dashboard.")
    parser.add_argument('--output', '-o', type=str, default='./evaluation_results_test', help="Output directory containing the CSV files")
    parser.add_argument('--interval', '-i', type=int, default=5, help="Polling interval in seconds")
    parser.add_argument('--url', '-u', type=str, default='https://test.lalithadithyan.dev/camino-places', help="URL of the C2 dashboard to post live updates")
    parser.add_argument('--total', type=int, default=2000, help="Total expected images per category")
    args = parser.parse_args()

    categories = ['SMALL', 'MEDIUM', 'LARGE']
    print(f"[*] Starting CSV progress monitor on directory: {args.output}")
    print(f"[*] Live updates URL: {args.url} | Interval: {args.interval}s")

    while True:
        payload = {}
        all_done = True

        for cat in categories:
            csv_path = os.path.join(args.output, f'metrics_{cat}.csv')
            count, avg_psnr, is_done = parse_csv(csv_path, args.total)
            
            payload[cat] = {
                "completions": count,
                "total": args.total,
                "psnr": float(round(avg_psnr, 4)),
                "done": is_done
            }
            
            if not is_done:
                all_done = False

        # Send live update payload as JSON POST
        send_to_dashboard(args.url, payload)

        if all_done:
            print("[*] All categories are marked as Done. Exiting monitor.")
            break

        time.sleep(args.interval)

if __name__ == '__main__':
    main()
