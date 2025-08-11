from vibenet.utils import load

df = load('data/fma_metadata/echonest.csv')
shuffled = df.sample(frac=1, random_state=42)

N = len(shuffled)
train_size = int(N * 0.85)
test_size = N - train_size
print("Train:", train_size, "Test:", test_size)

train_df = shuffled[:train_size]
test_df = shuffled[train_size:]

train_df.to_csv("train.csv")
test_df.to_csv("test.csv")