import time
import numpy as np
from pynput import keyboard
from simple_env import SimplePathFollowingEnv

class ManualControl:
    def __init__(self, env):
        self.env = env
        self.current_action = np.array([0.0, 0.0])  # [linear_velocity, angular_velocity]
        self.action_map = {
            keyboard.Key.up: [0.1, 0.0],     # Acelera para frente
            keyboard.Key.down: [-0.1, 0.0],  # Ré
            keyboard.Key.left: [0.0, 0.5],   # Gira para esquerda
            keyboard.Key.right: [0.0, -0.5], # Gira para direita
            keyboard.Key.space: [0.0, 0.0]   # Para o robô
        }
        self.listener = None
        self.running = True

    def on_press(self, key):
        '''
        When a key is pressed, we can change the action.
        '''
        if key in self.action_map:
            # Soma a ação (permite combinar teclas)
            self.current_action += self.action_map[key]
            # print(f"Action: {self.current_action}")

    def on_release(self, key):
        '''
        When a key is released, we can either stop the action or keep it.
        '''
        if key == keyboard.Key.esc:
            # Tecla ESC para sair
            self.running = False
            return False
        elif key in self.action_map:
            # Subtrai a ação quando a tecla é solta
            self.current_action -= self.action_map[key]

    def start(self):
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()

        # Resetar o ambiente
        self.env.reset()
        
        # Loop principal de controle
        while self.running:
            # Executar a ação atual
            _, _, terminated, truncated, _ = self.env.step(self.current_action)
            
            # Renderizar o ambiente
            self.env.render()
            
            # Pequeno delay para evitar uso excessivo da CPU
            time.sleep(0.05)
            
            # Verificar se o episódio terminou
            if terminated or truncated:
                self.env.reset()

        # Encerrar o listener quando sair
        self.listener.stop()

if __name__ == "__main__":
    env = SimplePathFollowingEnv()
    controller = ManualControl(env)
    controller.start()