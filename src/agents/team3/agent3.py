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
        self.fig = plt.figure(figsize=(16, 10))
        self.ax1 = plt.subplot(2, 3, 1)
        self.ax2 = plt.subplot(2, 3, 2)
        self.ax3 = plt.subplot(2, 3, 3)
        self.ax4 = plt.subplot(2, 3, 4)  # Direction
        self.ax5 = plt.subplot(2, 3, 5)  # Nouveau: Carte locale des nœuds
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
        if len(self.history) < 2:
            return
        
        frames = [h["frame"] for h in self.history]
        x_moyens = [h["x_moyen"] for h in self.history]
        curvatures = [h["curvature"] for h in self.history]
        speeds = [h["speed"] for h in self.history]
        accelerations = [h["acceleration"] for h in self.history]
        brakes = [1 if h["brake"] else 0 for h in self.history]
        
        # === GRAPHIQUE 1: Steering (x_moyen) ===
        self.ax1.clear()
        self.ax1.plot(frames, x_moyens, label="x_moyen (averaged)", color="blue", linewidth=2)
        self.ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        self.ax1.fill_between(frames, x_moyens, alpha=0.3, color="blue")
        self.ax1.set_ylabel("Steering Input (x_moyen)", fontweight='bold')
        self.ax1.set_title("Steering Control", fontweight='bold')
        self.ax1.legend(loc='upper left')
        self.ax1.grid(True, alpha=0.3)
        
        # === GRAPHIQUE 2: Curvature & Speed ===
        self.ax2.clear()
        ax2_twin = self.ax2.twinx()
        
        line1 = self.ax2.plot(frames, curvatures, label="Curvature", color="orange", linewidth=2)
        line2 = ax2_twin.plot(frames, speeds, label="Speed", color="red", linewidth=2)
        
        self.ax2.set_ylabel("Curvature", fontweight='bold', color="orange")
        ax2_twin.set_ylabel("Speed (m/s)", fontweight='bold', color="red")
        self.ax2.set_title("Track Curvature & Velocity", fontweight='bold')
        self.ax2.tick_params(axis='y', labelcolor='orange')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        self.ax2.grid(True, alpha=0.3)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        self.ax2.legend(lines, labels, loc='upper left')
        
        # === GRAPHIQUE 3: Accélération & Freinage ===
        self.ax3.clear()
        self.ax3.plot(frames, accelerations, label="Acceleration", color="green", linewidth=2, marker='o', markersize=3)
        self.ax3.scatter(frames, brakes, label="Brake (on/off)", color="red", s=20, alpha=0.6)
        self.ax3.set_ylabel("Value", fontweight='bold')
        self.ax3.set_xlabel("Frame", fontweight='bold')
        self.ax3.set_title("Acceleration & Brake Control", fontweight='bold')
        self.ax3.legend(loc='upper left')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.set_ylim(-0.2, 1.2)
        
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
