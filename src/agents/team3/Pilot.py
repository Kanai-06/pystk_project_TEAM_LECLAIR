import numpy as np

from utils.track_utils import compute_curvature, compute_slope

from omegaconf import OmegaConf

cfg = OmegaConf.load("../agents/team3/config.yml")


class Pilot():
    def choose_action(self, obs):
        targets = obs["paths_end"][:cfg.drift.node_look_ahead] #returne les vectors [x,y,z] des plus proches nœuds
        drift = False
        
        x_moyen = 0
        for target in targets:
            x = target[0] # prendre l'x
            x_moyen += x
        
        # x = obs["paths_end"][0][0]
        x = x_moyen / cfg.drift.node_look_ahead
        
        curvature = compute_curvature(targets) if targets is not None else 0
        # print(curvature)

        # vitesse / accélération / nitro
        energy = obs["energy"][0]
        nitro = False
        if abs(x)/20 > cfg.steering.sharp_turn_threshold and obs["distance_down_track"] > 5.0:
            acceleration = cfg.acceleration.sharp_turn
            brake = True
        elif energy > cfg.nitro.min_energy and abs(x)/20 < cfg.steering.straight_threshold:
            acceleration = 1
            brake = False
            nitro = True
        else:
            acceleration = 1
            brake = False

        # anti-blocage
        rescue = False
        speed = obs["velocity"][2]
        if speed < cfg.speed.slow_speed_threshold and obs["distance_down_track"] > 5.0:
            self.time_blocked += 1
            if self.time_blocked > cfg.speed.unblock_time:
                rescue = True
                acceleration = 0.0
                brake = True
                x = -x

        if self.time_blocked >= cfg.speed.reset_block_time:
            self.time_blocked = 0

        boost = obs["attachment"]
        use_fire = False
        if boost is not None:
            if obs["items_type"][0] == cfg.fire.enemy_type and boost == cfg.fire.required_attachment:
                use_fire = True

        # === LOGGING DES DONNÉES ===
        self.history.append({
            "frame": self.frame_count,
            "x_moyen": x_moyen,
            "x": x,
            "curvature": curvature,
            "speed": speed,
            "acceleration": acceleration,
            "brake": brake,
            "drift": drift,
        })
        self.frame_count += 1
        
        # Mettre à jour le graphique tous les 20 frames
        # if self.frame_count % 20 == 0:
        self.update_plot(paths_end=obs["paths_end"])

        action = {
            "acceleration": acceleration,
            "steer": x,
            "brake": brake,
            "drift": drift,
            "nitro": nitro,
            "rescue": rescue,
            "fire": use_fire,
        }
        return action
