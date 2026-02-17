from agents.kart_agent import KartAgent
from agents.team3.FireItems import FireItems
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

class Agent3(KartAgent):
    def __init__(self, env, path_lookahead=3):
        super().__init__(env)
        self.path_lookahead = path_lookahead
        self.agent_positions = []
        self.obs = None
        self.isEnd = False
        self.name = "TEAM L'ÉCLAIR"
        self.time_blocked = 0
        
        # === Initialisation pour Pilot (visualisation) ===
        self.frame_count = 0
        self.history = deque(maxlen=200)  # Garder les 200 derniers frames
        
        # Créer la figure matplotlib une seule fois
        plt.ion()  # Mode interactif
        self.fig = plt.figure(figsize=(7, 5))
        self.ax4 = plt.subplot(1, 2, 1)  # Direction
        self.ax5 = plt.subplot(1, 2, 2)  # Nouveau: Carte locale des nœuds
        self.fig.suptitle("Pilot Input Analysis - TEAM L'ÉCLAIR", fontsize=14, fontweight='bold')
        plt.tight_layout()

    def reset(self):
        self.obs, _ = self.env.reset()
        self.agent_positions = [] 

    def endOfTrack(self):
        return self.isEnd
    
    def choose_action(self, obs):
        action = FireItems.choose_action(self, obs)
        return action
    
    def update_plot(self, paths_end=None):
        """Mettre à jour les graphiques en temps réel."""
        # === GRAPHIQUE 4: Affichage de Direction (flèche) ===
        self.ax4.clear()
        self.ax4.set_xlim(-2, 2)
        self.ax4.set_ylim(-2, 2)
        self.ax4.set_aspect('equal')
        
        # Obtenir la dernière valeur de x_moyen
        if len(self.history) > 0:
            last_x_moyen = self.history[-1]["x_moyen"]
            # Normaliser entre -1 et 1
            normalized_x = max(-1, min(1, last_x_moyen / 20))
            
            # Dessiner un cercle de référence
            circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
            self.ax4.add_patch(circle)
            
            # Dessiner la flèche de direction
            # Les coordonnées: X horizontalement (gauche-droite), Y vers le haut (direction forward)
            arrow_length = 1.2
            self.ax4.arrow(0, 0, normalized_x * arrow_length, 1 * arrow_length, 
                          head_width=0.25, head_length=0.2, fc='blue', ec='blue', linewidth=3)
            
            # Ajouter les labels
            self.ax4.text(0, -1.6, f"x_moyen: {last_x_moyen:.2f}", 
                         ha='center', fontsize=12, fontweight='bold')
            self.ax4.text(-1.5, 1.5, "LEFT", ha='center', fontsize=10, color='red')
            self.ax4.text(1.5, 1.5, "RIGHT", ha='center', fontsize=10, color='red')
            self.ax4.text(0, 1.7, "FORWARD", ha='center', fontsize=10, color='green', fontweight='bold')
        
        self.ax4.set_xlabel("Steering Direction", fontweight='bold')
        self.ax4.set_title("Real-time Steering Direction", fontweight='bold')
        self.ax4.set_xticks([])
        self.ax4.set_yticks([])
        
        # === GRAPHIQUE 5: Carte locale des nœuds (paths_end) ===
        self.ax5.clear()
        self.ax5.set_aspect('equal')
        
        # Afficher les limites de la piste (estimée)
        self.ax5.set_xlim(-10, 10)
        self.ax5.set_ylim(-2, 30)
        
        if paths_end is not None and len(paths_end) > 0:
            # Extraire les coordonnées x, z (vue de dessus)
            paths_array = np.array(paths_end)
            x_coords = paths_array[:, 0]  # X latéral
            z_coords = paths_array[:, 2]  # Z (forward)
            
            # Afficher les nœuds futurs
            self.ax5.scatter(x_coords, z_coords, c=range(len(x_coords)), 
                            cmap='viridis', s=100, alpha=0.8, edgecolors='black', linewidth=2)
            
            # Afficher les numéros des nœuds
            for i, (x, z) in enumerate(zip(x_coords, z_coords)):
                self.ax5.text(x, z, str(i), ha='center', va='center', 
                             fontsize=8, fontweight='bold', color='white')
            
            # Tracer une ligne entre les nœuds
            self.ax5.plot(x_coords, z_coords, 'b--', alpha=0.5, linewidth=1)
        
        # Le kart est au centre (0, 0)
        self.ax5.scatter(0, 0, marker='o', s=300, c='red', label='Kart Position', 
                        edgecolors='darkred', linewidth=2, zorder=5)
        self.ax5.arrow(0, 0, 0, 3, head_width=0.5, head_length=0.5, 
                      fc='red', ec='red', linewidth=2, zorder=5)  # Direction forward
        
        self.ax5.set_xlabel("Lateral Position (X)", fontweight='bold')
        self.ax5.set_ylabel("Forward Position (Z)", fontweight='bold')
        self.ax5.set_title("Local Track Map - Paths End View", fontweight='bold')
        self.ax5.grid(True, alpha=0.3)
        self.ax5.legend(loc='upper right')
        
        plt.pause(0.001)
