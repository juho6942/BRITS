import os

data_dir = r"c:\Schoolwork\BIOMED\BRITS\Data\training_setA\training_setA"
files = [f for f in os.listdir(data_dir) if f.endswith('.psv')]

count = 0
for idx, f in enumerate(files):
    if idx % 5000 == 0:
        print(f"Scanned {idx} files...")
    with open(os.path.join(data_dir, f), 'r') as fp:
        # Count lines. Subtract 1 for header.
        lines = sum(1 for _ in fp)
        seq_len = lines - 1
        if seq_len > 50:
            count += 1

print(f"Number of patients with > 50 hours of data: {count}")
