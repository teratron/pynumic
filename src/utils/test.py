hidden_layers = [5, 3, 7, 4]

n = 0
for i in hidden_layers:
    i += 2
    n += 1
    print(i)

print(hidden_layers, n)

for j in range(len(hidden_layers)):
    hidden_layers[j] += 2
    print(hidden_layers[j])

print(hidden_layers)
