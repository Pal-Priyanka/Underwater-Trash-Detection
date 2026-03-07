
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def generate_diagram():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Define nodes: (x, y, label)
    nodes = {
        'Data': (1, 5),
        'Cleaning': (3, 5),
        'EDA': (3, 2),
        'Augmentation': (5, 5),
        'YOLO Branch': (7.5, 7),
        'DETR Branch': (7.5, 3),
        'Evaluation': (9.5, 5),
        'Frontend App': (11, 5)
    }

    # Draw nodes as boxes
    for name, (x, y) in nodes.items():
        color = '#008080' if 'Branch' in name else '#005f5f'
        ax.add_patch(patches.FancyBboxPatch((x-0.6, y-0.4), 1.2, 0.8, boxstyle="round,pad=0.1", 
                                          facecolor=color, edgecolor='white', linewidth=2))
        ax.text(x, y, name, ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    # Draw connections
    arrows = [
        ('Data', 'Cleaning'),
        ('Cleaning', 'EDA'),
        ('Cleaning', 'Augmentation'),
        ('Augmentation', 'YOLO Branch'),
        ('Augmentation', 'DETR Branch'),
        ('YOLO Branch', 'Evaluation'),
        ('DETR Branch', 'Evaluation'),
        ('Evaluation', 'Frontend App')
    ]

    for start, end in arrows:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        # Adjust endpoints for box boundaries
        dx = x2 - x1
        dy = y2 - y1
        ax.annotate('', xy=(x2-0.7 if dx>0 else x2+0.7, y2), 
                    xytext=(x1+0.7 if dx>0 else x1-0.7, y1),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#333333'))

    plt.title('Underwater Trash Detection — System Architecture', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('screenshots/architecture_diagram.png', dpi=300, bbox_inches='tight')
    print("Architecture diagram saved to screenshots/architecture_diagram.png")

if __name__ == "__main__":
    generate_diagram()
