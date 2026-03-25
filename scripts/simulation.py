#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Módulo que implementa a classe que gerencia a simulação
'''

import pygame
import pygame.gfxdraw
import numpy as np
import tugger
from bzpath import BzPath
import bezier
import os
import pickle

DEFAULT_BEZIER_FILENAME = 'bezier.bz'
DEFAULT_RAW_FILENAME = 'raw.dat'
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
WINDOW_CAPTION = 'PyTugger'
BACKGROUND_COLOR = 128, 128, 128
CAR_COLOR = 255, 255, 255
TRAJ_COLOR = 0, 0, 0
CURR_TRAJ_COLOR = 255, 174, 66
EDIT_TRAJ_COLOR = 255, 0, 0
CURR_DMP_GOAL_COLOR = 0, 255, 0
POINTS_LOOKHEAD_COLOR = 106, 90, 205
RAWPATH_COLOR = 0, 174, 174
KP_SCALE = 0.1
KP_CENTER = 0.2
DEFAULT_SCALE = 40.0
LOOKAHEAD_ANGLESPAN = np.pi / 2.0
LOOKAHEAD_TOTALPATHS = 100
LOOKAHEAD_DISTBTWPOINTS = 0.3
LOOKAHEAD_POINTSPERPATH = 20
LOOKAHEAD_SIMSTEPS = 10
LOOKAHEAD_COLOR = 100, 100, 100
LOOKAHEAD_SELECTED_COLOR = 246, 0, 255
DEFAULT, FOLLOWING = range(2)
POINT_DRAW_LOOKAHEAD_SIZE = 5.0
POINT_DRAW_TRAJ_SIZE = 5.0
FOLLOW_STEP_LENGTH = 0.01
KP_STEERING = 1.0
KP_WHEEL = 5.0
POINTS_TRAJ_COLOR = 246, 0, 255
POINTS_TEST_COLOR = 50, 60, 85

N_CARTS = 4

'''
- Remover o desenho das DMP's ✓
- Remover seleção de DMP's ✓
- Trazer para simulation o desenho das curvas de bezier
    - Ter load dentro de PzPath
- Trazer para simulation ativação de PathFollow
'''


class Simulation:
    '''
    Essa classe implementa a simulação do comboio logístico,
    com visualização no PyGame. Essa classe está aqui pois
    ainda está dependendo do PyGame. Fica pendente limpar ela
    e mover para o módulo tugger.
    '''
    def __init__(self, filename=None,
                 screen_width=SCREEN_WIDTH,
                 screen_height=SCREEN_HEIGHT,
                 window_caption=WINDOW_CAPTION):

        # Inicializa o ambiente do pygame
        pygame.init()

        # Reads the filename for the bezier curve
        if filename is not None:
            self.filename = filename
        else:
            self.filename = DEFAULT_BEZIER_FILENAME

        # Largura e altura da janela, em pixels
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Define o título da janela
        pygame.display.set_caption(window_caption)

        # Cria a janela da simulação
        self.screen = pygame.display.set_mode(
                                          [self.screen_width,
                                           self.screen_height])

        # Cor de fundo
        self.screen.fill(BACKGROUND_COLOR)

        # Cria o timer para controlar o framerate
        self.clock = pygame.time.Clock()

        # A simulação vai encerrar o laço principal
        # quando essa flag for verdadeira.
        self.exit = False

        # Lista de objetos a serem simulados
        self.objs = list()

        # Cria os objetos
        self.create_objects()

        # Tempo de simulação
        self.dt = 0.01

        # Controle nulo
        self.u = np.array([0.0, 0.0]).T

        self.converted_points = []
        self.converted_points_follow = []

        # Objetivo
        self.goal = np.array([[0.0, 0.0]])

        # Flag para controle PD
        self.pd_controller = False

        # Flag para gravação de trajetórias
        self.is_recording = False

        self.path_points = list()

        self.curr_dist = 0.0

        # Status do PathFollow
        self.status = DEFAULT

        # If the filename exists, load the bezier curves
        # from there
        if os.path.isfile(filename):
            file = open(self.filename, 'rb')
            self.bzcurves = pickle.load(file)
        else:
            # If the filename does not exist, will
            # create new bezier curves object. First,
            # however, we test to see if we can create
            # the file (to avoid letting the user edit
            # a whole path only to find out he cannot
            # save because of some error in the
            # provided filename).
            file = open(self.filename, 'wb')
            self.bzcurves = BzPath()

        # Flag para indicar que deve seguir
        # uma trajetória
        self.follow_traj = False
        self.traj_idx = 0

        # Controles da janela
        self.scale = DEFAULT_SCALE
        self.centerx = 0.0
        self.centery = 0.0

        self._scale = self.scale
        self._centerx = self.centerx
        self._centery = self.centery

        # Generate the steering cloud of points
        self.lookahead = dict()
        self.generate_lookahead()

        # This will record raw coordinates if recording is true
        self.raw_path = list()

    def create_objects(self):
        '''
        Creates the simulated objects
        '''
        self.train = tugger.Train(N_CARTS)

    def handle_events(self):
        '''
        Detecta e trata os eventos interativos
        do PyGame.
        '''

        # Examina os eventos acumulados na fila
        # de eventos, um a um
        for event in pygame.event.get():
            # Evento de encerrar a aplicação, por exemplo
            # fechando a janela.
            if event.type == pygame.QUIT:
                self.exit = True

            # Eventos de cliques do mouse
            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                self.goal = tugger.pygame2coords(np.array([[mouse_x,
                                                            mouse_y]]),
                                                 SCREEN_WIDTH, SCREEN_HEIGHT,
                                                 self._centerx, self._centery,
                                                 self._scale)
                self.pd_controller = True

            # Eventos de pressionamento de teclas
            elif event.type == pygame.KEYUP:
                # Tecla ESC encerra a aplicação
                # PathFollow control
                if event.key == pygame.K_KP0 or event.key == pygame.K_0:
                    # Start/stop following path
                    if self.status == DEFAULT:
                        self.status = FOLLOWING
                    elif self.status == FOLLOWING:
                        self.status = DEFAULT
                elif event.key == pygame.K_ESCAPE:
                    self.exit = True
                # Controles para dirigir o carro
                elif event.key == pygame.K_UP:
                    self.pd_controller = False
                    self.u[0] += 10.0
                elif event.key == pygame.K_DOWN:
                    self.pd_controller = False
                    self.u[0] -= 10.0
                elif event.key == pygame.K_RIGHT:
                    self.pd_controller = False
                    self.u[1] -= 10.0
                elif event.key == pygame.K_LEFT:
                    self.pd_controller = False
                    self.u[1] += 10.0
                elif event.key == pygame.K_SPACE:
                    if self.is_recording:
                        f = open(DEFAULT_RAW_FILENAME, 'wb')
                        pickle.dump(self.raw_path, f)
                        f.close()
                        self.is_recording = False
                    self.pd_controller = False
                    self.follow_traj = False
                    self.u[0] = 0.0
                    self.u[1] = -self.train.get_steer_angle()
                elif event.key == pygame.K_g:
                    # Switch on/off the PD controller
                    self.pd_controller = not self.pd_controller
                # Controles do zoom da janela
                elif event.key == pygame.K_q:
                    self.scale *= 2.0
                    self.centerx, self.centery = self.train.get_coord()
                elif event.key == pygame.K_e:
                    self.scale /= 2.0
                    self.centerx, self.centery = self.train.get_coord()
                elif event.key == pygame.K_w:
                    self.scale = DEFAULT_SCALE
                    self.centerx, self.centery = self.train.get_coord()
                elif event.key == pygame.K_r:
                    self.raw_path = list()
                    self.is_recording = True

    def draw(self):
        '''
        Desenha os objetos
        '''

        # Desenha a trajetória crua, se houver
        if len(self.raw_path) > 1:
            coord = tugger.coords2pygame(np.array(self.raw_path),
                                         SCREEN_WIDTH, SCREEN_HEIGHT,
                                         self._centerx, self._centery,
                                         self._scale)
            pygame.draw.aalines(self.screen, RAWPATH_COLOR, False, coord)

        # Desenha as trajetórias
        ptot = len(self.bzcurves.path_points)
        path_points = tugger.coords2pygame(np.array(self.bzcurves.path_points),
                                           SCREEN_WIDTH, SCREEN_HEIGHT,
                                           self._centerx, self._centery,
                                           self._scale)
        ctot = len(self.bzcurves.ctrl_points)
        ctrl_points = tugger.coords2pygame(np.array(self.bzcurves.ctrl_points),
                                           SCREEN_WIDTH, SCREEN_HEIGHT,
                                           self._centerx, self._centery,
                                           self._scale)
        if ptot > 2:
            for i in range(ptot-1):
                p1 = path_points[i]
                p2 = path_points[i+1]
                if 2*i+1 < ctot:
                    c1 = ctrl_points[2*i]
                    c2 = ctrl_points[2*i+1]
                    p1x, p1y = p1
                    p2x, p2y = p2
                    c1x, c1y = c1
                    c2x, c2y = c2
                    nodes = np.asfortranarray([[p1x, c1x, c2x, p2x],
                                               [p1y, c1y, c2y, p2y]])
                    curve = bezier.Curve.from_nodes(nodes)
                    vals = np.linspace(0.0, 1.0, num=20)
                    p = curve.evaluate_multi(vals).astype(int)
                    pygame.draw.aalines(self.screen, pygame.Color('#ffe400'),
                                        False, p.T.tolist())

        '''# Draw Path Points

        for point in path_points:
            self.converted_points.append(np.array([[point[0]], [point[1]]]))
        #print(converted_points)

        for i in range(ptot):
            for point in self.converted_points:
                if point is not None:
                    px, py = self.converted_points[i]
                    width = POINT_DRAW_TRAJ_SIZE
                    height = POINT_DRAW_TRAJ_SIZE
                    color = pygame.Color(POINTS_LOOKHEAD_COLOR)
                    left = int(px - POINT_DRAW_TRAJ_SIZE/2.0)
                    top = int(py - POINT_DRAW_TRAJ_SIZE/2.0)
                    pygame.draw.rect(self.screen, color,
                                    pygame.Rect(left, top, width, height))'''

        # Desenha o lookahead

        for _, points in self.lookahead.items():
            points = tugger.coords2pygame(points,
                                          SCREEN_WIDTH, SCREEN_HEIGHT,
                                          self._centerx, self._centery,
                                          self._scale)
            pygame.draw.aalines(self.screen, LOOKAHEAD_COLOR, False, points)

        # Desenha pontos no lookahead

        points = tugger.coords2pygame(np.array(self.min_path),
                                      SCREEN_WIDTH, SCREEN_HEIGHT,
                                      self._centerx, self._centery,
                                      self._scale)

        for point in points:
            width = POINT_DRAW_LOOKAHEAD_SIZE
            height = POINT_DRAW_LOOKAHEAD_SIZE
            px, py = point
            color = pygame.Color(POINTS_LOOKHEAD_COLOR)
            left = int(px - POINT_DRAW_LOOKAHEAD_SIZE/2.0)
            top = int(py - POINT_DRAW_LOOKAHEAD_SIZE/2.0)
            pygame.draw.rect(self.screen, color,
                             pygame.Rect(left, top, width, height))

        # Desenha os pontos na trajetória

        look_path_points = \
            tugger.coords2pygame(np.array(self.path_points),
                                 SCREEN_WIDTH, SCREEN_HEIGHT,
                                 self._centerx, self._centery,
                                 self._scale)

        for point in look_path_points:
            width = POINT_DRAW_TRAJ_SIZE
            height = POINT_DRAW_TRAJ_SIZE
            px, py = point
            color = pygame.Color(POINTS_TRAJ_COLOR)
            left = int(px - POINT_DRAW_TRAJ_SIZE/2.0)
            top = int(py - POINT_DRAW_TRAJ_SIZE/2.0)
            pygame.draw.rect(self.screen, color,
                             pygame.Rect(left, top, width, height))

        # Teste distâncias
        '''
        for point in look_path_points:
            if point is not None:
                px_follow, py_follow = point
                px_lookahead, py_lookahead = points[i]

                point_x = px_follow - px_lookahead
                point_y = py_follow - py_lookahead

                left = int(point_x - POINT_DRAW_TRAJ_SIZE/2.0)
                top = int(point_y - POINT_DRAW_TRAJ_SIZE/2.0)
                pygame.draw.rect(self.screen, POINTS_TEST_COLOR,
                                 pygame.Rect(left, top, width, height))'''

        # Desenha o veículo

        polys, circles = self.train.get_geometries()

        for poly in polys:
            coord = tugger.coords2pygame(poly,
                                         SCREEN_WIDTH, SCREEN_HEIGHT,
                                         self._centerx, self._centery,
                                         self._scale)
            pygame.draw.aalines(self.screen, CAR_COLOR, True, coord)

        for circle in circles:
            c = tugger.coords2pygame(circle,
                                     SCREEN_WIDTH, SCREEN_HEIGHT,
                                     self._centerx, self._centery, self._scale)
            for x, y in c:
                pygame.gfxdraw.aacircle(self.screen, int(x), int(y),
                                        int(0.1*self._scale), CAR_COLOR)

        goal = tugger.coords2pygame(np.array(self.goal),
                                    SCREEN_WIDTH, SCREEN_HEIGHT,
                                    self._centerx, self._centery, self._scale)

        color = CURR_DMP_GOAL_COLOR
        gy, gx = goal[0]
        pygame.draw.aalines(self.screen,
                            color, True, [(gy-int(0.2*self._scale),
                                           gx+int(0.2*self._scale)),
                                          (gy+int(0.2*self._scale),
                                           gx-int(0.2*self._scale))])
        pygame.draw.aalines(self.screen,
                            color, True, [(gy-int(0.2*self._scale),
                                           gx-int(0.2*self._scale)),
                                          (gy+int(0.2*self._scale),
                                           gx+int(0.2*self._scale))])

        self._scale += KP_SCALE * (self.scale - self._scale)
        self._centerx += KP_CENTER * (self.centerx - self._centerx)
        self._centery += KP_CENTER * (self.centery - self._centery)

    def update(self):
        '''
        Simulação dinâmica
        '''

        tractor_x, tractor_y, tractor_steer, tractor_angle = \
            self.train.get_state()[:4]

        if self.is_recording:
            self.raw_path.append((tractor_x, tractor_y))

        self.lookahead = dict()
        for steer_angle, points in self.lookahead_ref.items():
            points = np.array(points)
            d = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
            angle = np.arctan2(points[:, 1], points[:, 0])
            xs = np.array(tractor_x +
                          d * np.cos(angle+tractor_angle)).reshape(-1, 1)
            ys = np.array(tractor_y +
                          d * np.sin(angle+tractor_angle)).reshape(-1, 1)
            points = np.concatenate((xs, ys), axis=1)
            self.lookahead[steer_angle] = points

        if self.status == FOLLOWING:
            self.curr_dist += FOLLOW_STEP_LENGTH
            self.path_points = \
                self.bzcurves.getNextCoords(self.curr_dist,
                                            LOOKAHEAD_POINTSPERPATH,
                                            LOOKAHEAD_DISTBTWPOINTS)
            if self.curr_dist > self.bzcurves.getLength() \
               - LOOKAHEAD_POINTSPERPATH*LOOKAHEAD_DISTBTWPOINTS:
                self.path_points = []
                self.curr_dist = 0.0
        elif self.status == DEFAULT:
            pass

        min_cost = None
        desired_angle = 0.0
        for angle, path in self.lookahead.items():
            cost = self.compare_paths(path, self.path_points)
            if min_cost is None or cost < min_cost:
                min_cost = cost
                desired_angle = angle
                self.min_path = path

        # Lê as coordenadas do rebocador
        # x, y = self.train.tractor.get_tug_coord()
        x, y = self.train.tractor.get_state()

        # Lê o vetor de estados atual do comboio
        q = self.train.get_state()

        # Calcula o Jacobiano atual do comboio
        j = self.train.get_jacobian()

        # Aplica controle PD para o ponto clicado na tela
        if self.pd_controller:
            gy, gx = self.goal[0]
            self.u = self.train.get_control_for_goal(gy, gx, self.dt)
        # Aplica controle PD para seguir trajetória
        elif self.status == FOLLOWING and \
                len(self.path_points) == LOOKAHEAD_POINTSPERPATH:
            x1 = self.path_points[0][0][0]
            y1 = self.path_points[0][1][0]
            x2, y2 = self.lookahead[desired_angle][0]
            dx = x2 - x1
            dy = y2 - y1
            dist_err = np.sqrt(dx**2 + dy**2)
            current_angle = q[2]
            steer_err = desired_angle - current_angle
            steer_vel = KP_STEERING * steer_err
            wheel_vel = KP_WHEEL * dist_err
            self.u = np.array([wheel_vel, steer_vel])

        q_dot = np.dot(j, self.u)
        q += q_dot * self.dt
        self.train.set_state(q)

        # Para facilitar dirigibilidade, depois de
        # atualizar, reseta a velocidade do angulo
        # de esterçamento.
        if not self.pd_controller:
            self.u[1] = 0.0

    def generate_lookahead(self, angle_span=LOOKAHEAD_ANGLESPAN,
                           total_paths=LOOKAHEAD_TOTALPATHS,
                           dist_btw_points=LOOKAHEAD_DISTBTWPOINTS,
                           points_per_path=LOOKAHEAD_POINTSPERPATH,
                           sim_steps=LOOKAHEAD_SIMSTEPS):
        '''
        Esta função calcula os pontos referentes a diferentes esterçamentos
        do rebocador, olhando sempre à frente, para decidir qual esterçamento
        melhor alinhará com o caminho a ser seguido.
        '''
        d = dict()
        # Cria um trem com zero reboques (temporário)
        train = tugger.Train(0)
        # Para cada esterçamento
        dt = 1.0 / sim_steps
        for steering_angle in np.linspace(-angle_span/2.0,
                                          angle_span/2.0, total_paths):
            # Reseta posição do trator e ajusta esterçamento fixo
            u = np.array([0.0, 0.0, steering_angle, 0.0]).T
            train.set_state(u)
            d[steering_angle] = list()
            # Para cada ponto
            for _ in range(points_per_path):
                # Grava coordenada atual
                x, y = train.tractor.get_state()
                d[steering_angle].append((x, y))
                # Avança distância definida
                for _ in range(sim_steps):
                    step_dist = dist_btw_points/sim_steps
                    u = np.array([(step_dist / train.tractor.tyre_radius) / dt,
                                  0.0])
                    q = train.get_state()
                    j = train.get_jacobian()
                    q_dot = np.dot(j, u)
                    q += q_dot*dt
                    train.set_state(q)
        self.lookahead_ref = d

    def compare_paths(self, path1, path2):
        '''
        Compare the distance between the respective
        points of two trajectories, in the same
        order as they appear in the list.
        '''
        total_cost = 0.0
        weight = LOOKAHEAD_POINTSPERPATH
        for point1, point2 in zip(path1, path2):
            x1, y1 = point1
            x2, y2 = point2
            dx = x2 - x1
            dy = y2 - y1
            cost = dx**2 + dy**2
            total_cost += cost*weight
            weight -= 1.0
        return total_cost

    def run(self):
        '''
        Laço principal do programa
        '''
        while not self.exit:
            self.screen.fill(BACKGROUND_COLOR)
            self.update()
            self.draw()
            pygame.display.update()
            self.handle_events()
            self.clock.tick(30)


if __name__ == '__main__':
    pass
