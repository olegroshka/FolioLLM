import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Title
ax.text(5, 9.6, 'KAN LoRA Modification', horizontalalignment='center', verticalalignment='center', fontsize=16, fontweight='bold')

# Base Layer
base_box = patches.FancyBboxPatch((4, 8.8), 2, 0.2, boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightblue')
ax.add_patch(base_box)
ax.text(5, 8.9, 'Base Layer\nInput (batch_size, seq_len, 1024)', horizontalalignment='center', verticalalignment='center', fontsize=10)

# KAN Layer (replacement for LoRA_A)
kan_box = patches.FancyBboxPatch((4, 2.5), 2, 5.5, boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgreen')
ax.add_patch(kan_box)
ax.text(5, 8.0, 'KAN Layer replacing LoRA_A', horizontalalignment='center', verticalalignment='center', fontsize=12)

# Underlying layers in KAN
kan_layers = [
    ('Linear 1 (1024 -> 128)', (5, 7.6)),
    ('ReLU 1', (5, 6.9)),
    ('Linear 2 (128 -> 64)', (5, 6.2)),
    ('ReLU 2', (5, 5.5)),
    ('Linear 3 (64 -> 4)', (5, 4.8)),
    ('ReLU 3', (5, 4.1)),
    ('Linear 4 (4 -> 16)', (5, 3.4)),
    ('Output', (5, 2.7))
]

for layer in kan_layers:
    kan_layer_box = patches.FancyBboxPatch((layer[1][0] - 0.5, layer[1][1] - 0.1), 1, 0.05, boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightyellow')
    ax.add_patch(kan_layer_box)
    ax.text(layer[1][0], layer[1][1], layer[0], horizontalalignment='center', verticalalignment='center', fontsize=10)
    if layer[0] != 'Linear 1 (1024 -> 128)':
        ax.annotate('', xy=(layer[1][0], layer[1][1]+0.35), xytext=(layer[1][0], layer[1][1]+0.55)) #, arrowprops=dict(arrowstyle='->'))

# LoRA_B Layer
lora_b_box = patches.FancyBboxPatch((4, 1.5), 2, 0.2, boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightblue')
ax.add_patch(lora_b_box)
ax.text(5, 1.6, 'LoRA_B Layer\nOutput (batch_size, seq_len, 16)', horizontalalignment='center', verticalalignment='center', fontsize=10)

# Arrows
arrowprops = dict(facecolor='black', arrowstyle='->')
ax.annotate('', xy=(5, 8.5), xytext=(5, 7.5)) #, arrowprops=arrowprops)
ax.annotate('', xy=(5, 4.2), xytext=(5, 2))#, arrowprops=arrowprops)

# LoRA matrix equations
ax.text(2, 8, r'$W = W_0 + \Delta W$', fontsize=12)
ax.text(2, 7.5, r'$\Delta W = A \cdot B$', fontsize=12)
ax.text(2, 7, r'$A = KAN(x)$', fontsize=12)

# KAN mathematical formulas
ax.text(2, 5, r'$KAN(x) = (\Phi_3 \circ \Phi_2 \circ \Phi_1)(x)$', fontsize=12)
ax.text(2, 4.5, r'$\Phi_q = \sum_{p=1}^{n} \phi_{q,p}(x_p)$', fontsize=12)
ax.text(2, 4, r'$\phi_{q,p}(x_p)$', fontsize=12)

ax.axis('off')
plt.tight_layout()
plt.show()
