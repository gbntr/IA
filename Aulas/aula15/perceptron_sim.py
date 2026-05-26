# -*- coding: utf-8 -*-
data = [
    ("Joao",  [1, 1, 0, 1], 1),
    ("Pedro", [0, 0, 1, 0], 0),
    ("Maria", [1, 1, 0, 0], 0),
    ("Jose",  [1, 0, 1, 1], 1),
    ("Ana",   [1, 0, 0, 1], 0),
    ("Leila", [0, 0, 1, 1], 1),
]

eta = 0.5
theta = -0.5
w = [0.0, 0.0, 0.0, 0.0]

for epoch in range(1, 21):
    total_errors = 0
    print(f"\n{'='*60}")
    print(f"EPOCA {epoch} | pesos iniciais: w={[round(v,2) for v in w]}")
    print(f"{'='*60}")
    for nome, x, target in data:
        net = sum(w[j]*x[j] for j in range(4))
        output = 1 if net >= theta else 0
        error = target - output
        label_t = "Doente" if target==1 else "Saudavel"
        label_o = "Doente" if output==1 else "Saudavel"
        print(f"  {nome:6s} | x={x} | net={net:+.2f} | saida={label_o} | alvo={label_t} | erro={error:+d}")
        if error != 0:
            total_errors += 1
            for j in range(4):
                w[j] = round(w[j] + eta * error * x[j], 4)
            print(f"         => atualiza pesos: w={[round(v,2) for v in w]}")
    print(f"\n  Erros epoca {epoch}: {total_errors} | pesos finais: w={[round(v,2) for v in w]}")
    if total_errors == 0:
        print(f"  *** Convergiu na epoca {epoch}! ***")
        break

print(f"\n{'='*60}")
print(f"PESOS FINAIS: w1(Febre)={w[0]}, w2(Enjoo)={w[1]}, w3(Manchas)={w[2]}, w4(Dores)={w[3]}")

print(f"\n{'='*60}")
print("TESTE DE NOVOS CASOS (limiar = -0.5)")
print(f"{'='*60}")
novos = [
    ("Luis",  [0, 0, 0, 1]),
    ("Laura", [1, 1, 1, 1]),
]
for nome, x in novos:
    net = sum(w[j]*x[j] for j in range(4))
    output = 1 if net >= theta else 0
    label = "Doente" if output==1 else "Saudavel"
    print(f"  {nome:6s} | x={x} | net={net:+.4f} | resultado: {label}")
