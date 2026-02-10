import numpy as np
from agents.kart_agent import KartAgent

class Agent3(KartAgent):
    def __init__(self, env, path_lookahead=3):
        super().__init__(env)
        self.path_lookahead = path_lookahead
        self.obs = None
        self.isEnd = False
        self.name = "Team L'éclair"
        self.time_blocked = 0

    def reset(self):
        self.obs, _ = self.env.reset()
        self.agent_positions = [] 

    def endOfTrack(self):
        return self.isEnd
    
    def choose_action(self, obs):
        target = np.array(obs["paths_end"][0]) #Récupère le vecteur du prochain noeud
        x = target[0] #Récupère la coordonnée x de ce vecteur afin de savoir dans quelle direction le kart doit se diriger
        nitro = False #Par défaut le kart n'utilise pas le nitro
        fire = False
        items_pos = np.array(obs["items_position"]) #Récupère en temps réel les positions de chaque items du circuit et converti en array
        items_type = np.array(obs["items_type"]) #Récupère en temps réel les types de chaque items sur le circuit et converti en array
        correct_steering = 0.0 #Corrige le pilotage si nécessaire
        brake_urgency = False #Permet de freiner en urgence si l'arrivée d'une peau de banane est imminente
        bananas_pos = items_pos[items_type==1] #Récupère uniquement la position des bananes
        dangerous_bananas = [] #Initie une liste des bananes dangereuses (bananes imminentes)
        for b_pos in bananas_pos: #Pour chaque bananes sur le circuit, récupère le vecteur un par un
            b_x, b_y, b_z = b_pos #Récupère les coordonnées de la banane et les places dans des variables
            if 0 < b_z < 30 and abs(b_x) < 2.0: #Si la position de la banane est de moins de 25 mètres et que la position de la banane n'est pas devant nous
                dangerous_bananas.append(b_pos) #Ajoute la position de la banane dans notre liste des bananes dangereuses
        if len(dangerous_bananas) > 0: #S'il y a au moins une bananes dangereuses dans notre liste
            array = np.array(dangerous_bananas) #On converti notre liste en array
            sorted_indices = np.argsort(array[:, 2]) #On trie les indices des bananes selon le plus proche jusqu'au plus loin de notre position
            closest_banana = array[sorted_indices[0]] 
            b_x, _, b_z = closest_banana #On récupère les coordonnées x et z de la banane la plus proche
            if b_z > 10: #Si la banane la plus proche est à plus de 10 mètres
            	sensitivity = 1.0 #On change la sensibilité légèrement
            else:
            	sensitivity= 3.0 #On change la sensibilité de manière plus fort
            if b_x > 0: #Si la banane la plus proche se situe à notre gauche
                correct_steering = -sensitivity #On tourne à droite
            else:
                correct_steering = sensitivity #On tourne à gauche
            if b_z < 10 and abs(b_x) < 1.0: #Si la banane la plus proche est à moins de 8 mètres
                brake_urgency = True #On freine d'urgence pour éviter de foncer dedans
        steer = x + correct_steering #On corrige notre steering
        if abs(steer) > 1.0 or brake_urgency: #Si gros virage ou on doit freiner d'urgence
            if brake_urgency:
            	acceleration = 0.0 #On arrête d'accelerer 
            else:
            	acceleration = 0.2 #On accélère légèrement
            if brake_urgency: 
            	brake = True 
            else:
            	brake = False
        else: #On est dans une ligne droite et pas de peau de banane en vue alors on accélère à fond et on utilise le nitro si disponible
            acceleration = 1.0
            nitro = True
            brake = False
        speed = obs["velocity"][2]
        if (speed < 0.1): #Si la vitesse est faible
            self.time_blocked += 1 
            if (self.time_blocked > 10): #Si on reste bloqué un petit moment alors on fait marche arrière pendant un petit moment
                acceleration = 0.0
                brake = True
                steer = -steer
        if (self.time_blocked == 20): #On arrête de faire marche arrière et on se remet sur le droit chemin
            self.time_blocked = 0 
        if (x<1.0):
        	fire = True
        action = {
            "acceleration": acceleration,
            "steer": steer,
            "brake": brake,
            "drift": False,
            "nitro": False,
            "rescue": False,
            "fire": fire,
        }
        return action
