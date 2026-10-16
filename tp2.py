import mdptoolbox, mdptoolbox.example, mdptoolbox.util, mdptoolbox.mdp, numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# P: probability transition matrices
# R: reward matrix

states = 5
actions = 2
STATES = ["LD", "LC", "CD", "CC", "O"]
ACTIONS = ["nettoyer", "charger"]

def robot_nettoyage(e, eps, pDust, rc, rd, costcrash, e_charge=0):
    # dimension: (actions, states, states)
    # e_charge: risque de panne en nettoyant depuis le chargeur (0 = sujet d'origine)
    P = np.array([[[(1-e)*eps, (1-e)*(1-eps), 0, 0, e],
                  [0, (1-e), 0, 0, e],
                  [(1-e_charge)*eps, (1-e_charge)*(1-eps), 0, 0, e_charge],
                  [0, (1-e_charge), 0, 0, e_charge],
                  [0, 0, 0, 0, 1]],
                  [[0, 0, 1, 0, 0],
                   [0, 0, pDust, (1-pDust), 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, pDust, (1-pDust), 0],
                   [0, 0, 0, 0, 1]]])
    # dimension: (states, actions)
    R = np.array([[rd, rd], [rc, rc], [rd, rd], [rc, rc], [costcrash, costcrash]])
    return(P,R)

# rewards (question B) et parametres fixes
rc, rd, costcrash = 3, 1, -1
eps = 0.1
gamma = 0.9

def solve(e, pDust, e_charge=0, costcrash=costcrash, gamma=gamma):
    P, R = robot_nettoyage(e, eps, pDust, rc, rd, costcrash, e_charge)
    mdptoolbox.util.check(P, R)
    vi = mdptoolbox.mdp.ValueIteration(P, R, gamma, epsilon=1e-6)
    vi.run()
    return vi.policy, vi.V

# question C : quelques cas
cases = [("e eleve", 0.3, 0.1),
         ("e faible", 0.05, 0.1),
         ("tres bonne batterie", 0.001, 0.1),
         ("salon se salit vite", 0.001, 0.9)]
for name, e, pDust in cases:
    policy, V = solve(e, pDust)
    print(f"{name:22s} e={e:<6} pDust={pDust:<4}",
          {s: ACTIONS[a] for s, a in zip(STATES[:4], policy)},
          np.round(V, 2))

# balayage de e et pDust
e_values = np.linspace(0, 0.3, 61)
pDust_values = np.linspace(0, 1, 51)

def policy_map(e_charge=0):
    # actions choisies en LD et LC pour chaque couple (pDust, e)
    grid = np.zeros((2, len(pDust_values), len(e_values)), dtype=int)
    for i, pDust in enumerate(pDust_values):
        for j, e in enumerate(e_values):
            policy, _ = solve(e, pDust, e_charge)
            grid[0, i, j], grid[1, i, j] = policy[0], policy[1]
    return grid

grid = policy_map()

# V en fonction de e (pDust fixe)
pDust_fixed = 0.1
V_curves = np.array([solve(e, pDust_fixed)[1] for e in e_values])

BLUE, ORANGE = "#2a78d6", "#eb6834"
cmap = ListedColormap([BLUE, ORANGE])  # 0 = nettoyer, 1 = charger

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for k, ax in enumerate(axes[:2]):
    ax.pcolormesh(e_values, pDust_values, grid[k], cmap=cmap, vmin=0, vmax=1, shading="nearest")
    ax.set_title(f"Action optimale en {STATES[k]}")
    ax.set_xlabel("e (probabilite de panne)")
    ax.set_ylabel("pDust")
axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE, label="nettoyer"),
                        plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="charger")],
               loc="upper right")

ax = axes[2]
ax.plot(e_values, V_curves[:, 0], color=BLUE, lw=2, label="V(LD)")
ax.plot(e_values, V_curves[:, 1], color=ORANGE, lw=2, label="V(LC)")
ax.set_title(f"Valeurs selon e (pDust = {pDust_fixed})")
ax.set_xlabel("e (probabilite de panne)")
ax.set_ylabel("V")
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig("politique_e_pDust.png", dpi=120)

# variante : panne possible aussi en partant du chargeur
e_charge = 0.02
grid_c = policy_map(e_charge)
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4.5))
for k, ax in enumerate(axes2):
    ax.pcolormesh(e_values, pDust_values, grid_c[k], cmap=cmap, vmin=0, vmax=1, shading="nearest")
    ax.set_title(f"Action optimale en {STATES[k]} (e_charge = {e_charge})")
    ax.set_xlabel("e (probabilite de panne)")
    ax.set_ylabel("pDust")
axes2[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE, label="nettoyer"),
                         plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="charger")],
                loc="upper right")
fig2.tight_layout()
fig2.savefig("politique_variante_e_charge.png", dpi=120)

# seuil de e a partir duquel le robot choisit charger (recherche par dichotomie)
def seuil_e(state, e_max=0.3, **kw):
    if solve(e_max, pDust_fixed, **kw)[0][state] == 0:
        return np.nan  # nettoyer meme pour e = e_max
    lo, hi = 0.0, e_max
    for _ in range(25):
        mid = (lo + hi) / 2
        if solve(mid, pDust_fixed, **kw)[0][state] == 0:
            lo = mid
        else:
            hi = mid
    return hi

costcrash_values = -np.logspace(-1, 3, 25)  # de -0.1 a -1000
gamma_values = np.linspace(0.5, 0.99, 25)

fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
for k, color in [(0, BLUE), (1, ORANGE)]:
    for ec, ls in [(0, "-"), (e_charge, "--")]:
        s = [seuil_e(k, e_charge=ec, costcrash=c) for c in costcrash_values]
        ax1.plot(-costcrash_values, s, color=color, ls=ls, lw=2,
                 label=f"{STATES[k]}, e_charge = {ec}")
    s = [seuil_e(k, gamma=g) for g in gamma_values]
    ax2.plot(gamma_values, s, color=color, lw=2, label=STATES[k])

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_title(f"Seuil de e selon costcrash (gamma = {gamma}, pDust = {pDust_fixed})")
ax1.set_xlabel("-costcrash (echelle log)")
ax1.set_ylabel("seuil de e (echelle log)")
ax1.grid(alpha=0.3, which="both")
ax1.legend()

ax2.set_yscale("log")
ax2.set_title(f"Seuil de e selon gamma (costcrash = {costcrash}, pDust = {pDust_fixed})")
ax2.set_xlabel("gamma")
ax2.set_ylabel("seuil de e (echelle log)")
ax2.grid(alpha=0.3, which="both")
ax2.legend()

fig3.tight_layout()
fig3.savefig("seuil_costcrash_gamma.png", dpi=120)
plt.show()
