#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Este módulo define classes e funções para simulação cinemática
de um comboio logístico.
'''

import numpy as np


def transform_coords(pos_x: float, 
                     pos_y: float, 
                     angle: float, 
                     coords: np.ndarray) -> np.ndarray:
    '''
    @brief: Relative position to global coordinates

    @param pos_x: Posição x do objeto
    @param pos_y: Posição y do objeto
    @param angle: Ângulo de rotação do objeto
    @param coords: Coordenadas a serem transformadas
    '''
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
    translation_vector = np.array([[pos_x, pos_y]]).T
    coords = np.dot(rot_matrix, coords.T) + translation_vector
    return coords.T


def get_rect_coords(pos_x: float, 
                    pos_y: float, 
                    width: float, 
                    height: float, 
                    angle: float) -> np.ndarray:
    '''
    Retorna as coordenadas dos quatro cantos do retângulo no
    espaço 2D no formato NumPy array. Retorna uma matrix
    4x2 com um par de coordenadas (x, y) em cada linha.
    '''

    # Cria as coordenadas do retângulo centrado em zero.
    coords = np.array([[-width/2.0,  height/2.0],
                       [width/2.0,  height/2.0],
                       [width/2.0, -height/2.0],
                       [-width/2.0, -height/2.0]])

    # Calcula a rotação e translação no espaço 2D
    coords = transform_coords(pos_x, 
                              pos_y, 
                              angle, 
                              coords)

    # No momento de retornar o resultado,
    # se desfaz a transposição
    return coords


def coords2pyplot(coords):
    '''
    Transforma as coordenadas de N pontos, passadas
    como um array Nx2 do NumPy, para duas listas de
    coordenadas xs, ys, para uso da função plot(xs, ys).
    '''

    # Transforma em lista simples do Python
    coords = list(coords)

    # Fecha o polígono repetindo a primeira coordenada
    # novamente ao final.
    coords.append(coords[0])

    # Separa as coordenadas x e y em listas separadas
    x_list, y_list = zip(*coords)

    return x_list, y_list


def coords2pygame(coords, window_width, window_height,
                  centerx, centery, scale):
    '''
    Transforma as coordenadas para uso no pygame
    '''

    if len(coords) == 0:
        return list()

    coords = np.array(coords)

    # Ajusta formato do array de coordenadas
    np.reshape(coords.astype(float), (-1, 2))

    # Ajusta para a coordenada 0, 0 ficar no centro
    # da tela e inverte y.
    coords[:, 1] = float(window_height)/2.0 - scale*(coords[:, 1] - float(centery))
    coords[:, 0] = float(window_width)/2.0 + scale*(coords[:, 0] - float(centerx))

    # Transforma em lista de inteiros
    # coords = list((coords).astype(int))
    coords = list((coords))

    return coords


def pygame2coords(coords, window_width, window_height,
                  centerx, centery, scale):
    '''
    Transforma as coordenadas do pygame para simulação
    '''

    # Ajusta formado do array de coordenadas
    np.reshape(coords.astype(float), (-1, 2))

    # Calcula a posição do pygame para aa simulação
    coords[:, 1] = float(centery) + (float(window_height)/2.0 - coords[:, 1]) / scale
    coords[:, 0] = float(centerx) + (coords[:, 0] - float(window_width)/2.0) / scale

    # Transforma em lista de inteiros
    # coords = list((coords).astype(int))
    coords = list((coords))

    return coords


class Cart:
    '''
    Define um carrinho que fará parte de um comboio, a ser
    puxado por um trator.
    '''

    # Dynamic variables
    x_pos = 0.0
    y_pos = 0.0
    angle = 0.0
    steer_angle = 0.0

    # Fixed parameters
    tyre_radius = 0.288
    tyre_width = 0.167
    wheelbase = 1.5 # Não encontrei o que altera
    width = 1.0
    front_drawbar = 1.0 # Não encontrei o que altera
    back_drawbar = 5 # Não encontrei o que altera
    margin = 0.12
    tugger = None
    next_cart = None

    def __init__(self, x=0.0, y=0.0, angle=0.0, steer_angle=0.0):
        '''
        Inicializa variáveis dinâmicas
        '''
        self.x = x
        self.y = y
        self.angle = angle
        self.steer_angle = steer_angle

    def get_tug_coord(self):
        '''
        Retorna as coordenadas do ponto de contato do rebocador
        '''
        tug_x = self.x + self.back_drawbar * np.sin(self.angle)
        tug_y = self.y - self.back_drawbar * np.cos(self.angle)
        return tug_x, tug_y

    def set_tugged_by(self, tugger):
        '''
        Define o objeto que deve puxar este reboque. Você não
        precisa chamar esta função, pois a função abaixo já
        o faz.
        '''
        self.tugger = tugger

    def set_next_cart(self, cart):
        '''
        Define o próximo reboque da cadeia. Esta função
        já chama a acima, reciprocamente.
        '''
        cart.set_tugged_by(self)
        self.next_cart = cart

    def update_tugs(self):
        '''
        Atualiza a posição de todos elementos do comboio
        que vêm após este, chamando a função recursivamente.
        '''
        if self.tugger is not None:
            xt, yt = self.tugger.get_tug_coord()
            self.x = xt + self.front_drawbar \
                * np.sin(self.angle + self.steer_angle) \
                + self.wheelbase * np.sin(self.angle)
            self.y = yt - self.front_drawbar \
                * np.cos(self.angle + self.steer_angle) \
                - self.wheelbase * np.cos(self.angle)
        if self.next_cart is not None:
            self.next_cart.update_tugs()

    def get_geometries(self):
        '''
        Retorna os polígonos fechados que serão desenhados. Essas funções
        aqui dizem respeito apenas à representação gráfica.
        '''
        # Calcula o ângulo de esterçamento de cada roda, considerando
        # as diferenças de angulação entre rodas internas e externas
        # ao fazer a curva
        alpha_l = np.arctan2(self.wheelbase,
                             -self.width
                             + self.wheelbase/(np.tan(self.steer_angle)
                                               + 0.00001))
        alpha_r = np.arctan2(self.wheelbase,
                             self.width
                             + self.wheelbase/(np.tan(self.steer_angle)
                                               + 0.00001))

        # Computa as coordenadas dos retângulos que vão representar as
        # quatro rodas vistas de cima.
        tyres = [get_rect_coords(-self.width/2.0, +self.wheelbase/2.0,
                                 self.tyre_width, self.tyre_radius*2.0,
                                 alpha_l),
                 get_rect_coords(+self.width/2.0, +self.wheelbase/2.0,
                                 self.tyre_width, self.tyre_radius*2.0,
                                 alpha_r),
                 get_rect_coords(-self.width/2.0, -self.wheelbase/2.0,
                                 self.tyre_width, self.tyre_radius*2.0,
                                 -alpha_l),
                 get_rect_coords(+self.width/2.0, -self.wheelbase/2.0,
                                 self.tyre_width, self.tyre_radius*2.0,
                                 -alpha_r)]

        # Front, rear, and longitudinal axes
        axis = [np.array([[-self.width/2.0, +self.wheelbase/2.0],
                          [+self.width/2.0, +self.wheelbase/2.0]]),
                np.array([[-self.width/2.0, -self.wheelbase/2.0],
                          [+self.width/2.0, -self.wheelbase/2.0]]),
                np.array([[0.0, +self.wheelbase/2.0],
                          [0.0, -self.wheelbase/2.0]])]

        # Corpo
        body = [get_rect_coords(0.0, 0.0,
                                self.width+self.margin,
                                self.wheelbase+2*self.tyre_radius+self.margin,
                                0.0)]

        # Cambões
        x0 = 0.0
        y0 = self.wheelbase/2.0
        x1 = self.front_drawbar*np.cos(self.steer_angle+np.pi/2.0)
        y1 = y0+self.front_drawbar*np.sin(self.steer_angle+np.pi/2.0)
        x2 = 0.0
        y2 = -self.wheelbase/2.0
        x3 = 0.0
        y3 = y2-self.back_drawbar
        drawbars = [np.array([[x0, y0], [x1, y1]]),
                    np.array([[x2, y2], [x3, y3]])]
        circles = [np.array([[x0, y0], [x3, y3]])]

        # Coordenadas do centro do reboque
        xc = self.x - (self.wheelbase/2.0) * np.sin(self.angle)
        yc = self.y + (self.wheelbase/2.0) * np.cos(self.angle)

        # All polygons
        polys = tyres + axis + drawbars + body
        for i, poly in enumerate(polys):
            polys[i] = transform_coords(xc, yc, self.angle, poly)
        for i, circle in enumerate(circles):
            circles[i] = transform_coords(xc, yc, self.angle, circle)

        return polys, circles


class Tractor:
    '''
    Define um trator estilo triciclo que puxará o comboio.
    '''
    # Dynamic variables
    x = 0.0
    y = 0.0
    angle = 0.0
    steer_angle = 0.0

    # Fixed parameters
    wheel_radius = 0.0775
    wheel_height = 0.055
    wheel_separation = 0.41
    base_len = 0.5
    base_width = 0.335
    base_height = 0.235
    back_drawbar = 1.0
    next_cart = None

    def __init__(self, x=0.0, y=0.0, steer_angle=0.0, angle=0.0):
        '''
        Inicializa variáveis dinâmicas
        '''
        self.angle = angle
        self.steer_angle = steer_angle
        self.x = x
        self.y = y

    def set_state(self, 
                  x: float, 
                  y: float, 
                  steer_angle: float, 
                  angle: float):
        '''
        @brief: Define the state of the tractor.
        '''
        self.angle = angle
        self.steer_angle = steer_angle
        self.x = x
        self.y = y

    def get_state(self):
        return self.x, self.y

    def get_tug_coord(self):
        '''
        Retorna as coordenadas da junta de conexão do reboque
        '''
        tug_x = self.x + self.back_drawbar * np.sin(self.angle)
        tug_y = self.y - self.back_drawbar * np.cos(self.angle)
        return tug_x, tug_y

    def set_next_cart(self, cart):
        '''
        Define o reboque que vai preso ao rebocador.
        '''
        cart.set_tugged_by(self)
        self.next_cart = cart

    def update_tugs(self):
        '''
        Atualiza a posição dos reboques do comboio,
        chamando essa função para cada elemento, de
        forma recursiva.
        '''
        if self.next_cart is not None:
            self.next_cart.update_tugs()

    def get_geometries(self):
        '''
        @brief: Return the closed polygons and circles to be drawn.
        '''

        # Compute the coords to represent the tyres from bottom view
        tyres = [get_rect_coords(-self.wheel_separation/2.0, 
                                 0.0,
                                 self.wheel_height, # Wrong place but work better here
                                 self.wheel_radius, # Wrong place but work better here
                                 0.0),
                 get_rect_coords(+self.wheel_separation/2.0, 
                                 0.0,
                                 self.wheel_height, # Wrong place but work better here
                                 self.wheel_radius, # Wrong place but work better here
                                 0.0)]
        
        # Fix the body. The center of the rectangle shoud be on 
        body = [get_rect_coords(0.0,
                                0.0,
                                self.base_width,
                                self.base_len,
                                0.0)]
        
        # rear_tyres

        # Drawbars
        x2 = 0.0
        y2 = 0.0
        x3 = 0.0
        y3 = y2 - self.back_drawbar
        drawbars = [np.array([[x2, y2], [x3, y3]])]
        circles = [np.array([[x3, y3]])]

        # Coordenadas do centro do trator
        xc = self.x
        yc = self.y

        # All polygons
        polys = tyres + body + drawbars
        for i, poly in enumerate(polys):
            polys[i] = transform_coords(xc, yc, self.angle, poly)
        for i, circle in enumerate(circles):
            circles[i] = transform_coords(xc, yc, self.angle, circle)
        return polys, circles


class Train:
    '''
    Comboio completo, com trator e carrinhos de reboque
    '''

    # Número de reboques no comboio
    # (esse número não inclui o trator)
    n = 4

    # Posição do centro do eixo traseiro
    # do trator
    x = 0.0
    y = 0.0

    # Goal
    x_goal = 0.0
    y_goal = 0.0

    # Parâmetros dos reboques
    cart_wheelbase = 1.5
    cart_front_drawbar = 1.0
    cart_back_drawbar = 0.5

    tractor = None

    def __init__(self, n=4):
        '''
        Inicializa um comboio com um trem e n reboques,
        total de n+1 elementos.
        '''
        # Inicializa as coordenadas na
        # posição do trator
        x = self.x
        y = self.y

        # Inicializa os erros para
        # buscar a posição alvo
        self.d_err = 0.0
        self.angle_err = 0.0
        self.d_err_dot = 0.0
        self.angle_err_dot = 0.0

        # Copia n
        self.n = n

        # Nosso comboio será uma lista de objetos
        # começando pelo trator, seguido dos respectivos
        # reboques, em ordem
        self.train = list()

        # Criamos o trator e colocamos na lista
        self.tractor = Tractor(x, y)
        self.train.append(self.tractor)

        # Calcula a próxima posição y para o primeiro
        # reboque
        y -= self.tractor.back_drawbar + \
            self.cart_front_drawbar + self.cart_wheelbase

        # Laço para criar n reboques
        for _ in range(self.n):

            # Cria um reboque
            cart = Cart(x, y)

            # Ajusta os parâmetros do reboque
            cart.wheelbase = self.cart_wheelbase
            cart.front_drawbar = self.cart_front_drawbar
            cart.back_drawbar = self.cart_back_drawbar

            # Conecta o reboque ao elemento da frente
            # que pode ser o trator ou outro reboque
            self.train[-1].set_next_cart(cart)

            # Adiciona esse reboque ao comboio
            self.train.append(cart)

            # Calcula a posição y do próximo reboque
            y -= cart.front_drawbar + cart.wheelbase + cart.back_drawbar

    def update_tugs(self):
        '''
        Atualiza a posição de todos elementos, respeitando
        a posição das conexões. Essa função é chamada de
        forma recursiva de um elemento para o próximo,
        começando aqui pelo trator.
        '''
        if len(self.train) > 0:
            self.train[0].update_tugs()

    def update_goal(self, y, x, dt=0.01):
        '''
        Atualiza o alvo para onde o comboio deve
        tentar seguir
        '''
        # Atualiza o objetivo
        self.x_goal = x
        self.y_goal = y

        # Calcula o erro Cartesiano
        #tractor_x, tractor_y = self.tractor.get_tug_coord()
        #x_err = self.x_goal - tractor_y
        #y_err = self.y_goal - tractor_x
        x_err = self.x_goal - self.tractor.y
        y_err = self.y_goal - self.tractor.x

        # Calcula o erro polar
        d_err_new = np.sqrt(x_err**2 + y_err**2)
        angle_err_new = np.arctan2(-y_err, x_err) \
            - self.tractor.angle - self.tractor.steer_angle

        # TODO: Preciso disso?
        # if d_err_new < 0.02:
        #    d_err_new = 0.0
        #    angle_err_new = 0.0

        # Calcula as derivadas
        self.d_err_dot = (d_err_new - self.d_err) / dt
        self.angle_err_dot = (angle_err_new - self.angle_err) / dt

        # Atualiza os erros polares
        self.d_err = d_err_new
        self.angle_err = angle_err_new

    def get_state(self):
        '''
        Retorna o vetor de estados q
        '''
        if len(self.train) > 0:
            q = [self.tractor.x, self.tractor.y,
                 self.tractor.steer_angle, self.tractor.angle]
            for cart in self.train[1:]:
                q += [cart.angle+cart.steer_angle, cart.angle]
        q = np.array(q).T
        return q

    def get_steer_angle(self):
        '''
        Retorna o esterçamento do trator
        '''
        return self.train[0].steer_angle

    def get_coord(self):
        '''
        Retorna as coordenadas do ponto de contato do rebocador
        no trator
        '''
        return self.train[0].get_tug_coord()

    def set_state(self, q):
        '''
        Seta o vetor de estados q
        '''
        q = q.T
        # Ajusta o estado do trator
        if len(self.train) > 0:
            tractor = self.train[0]
            x, y, steer_angle, angle = q[:4]
            tractor.set_state(x, y, steer_angle, angle)
            # Ajusta o estado de cada reboque
            for i, cart in enumerate(self.train[1:]):
                cart.angle = q[4 + 2*i + 1]
                cart.steer_angle = q[4 + 2*i] - cart.angle
        # Atualiza posições
        self.update_tugs()

    def get_jacobian(self):
        '''
        Retorna o Jacobiano do sistema. Aqui
        eu segui o artigo https://doi.org/10.2507/IJSIMM20-2-550
        Foram feitos apenas alguns pequenos ajustes.
        '''

        # Iniciamos com uma lista, pois fica mais simples
        # de adicionar elementos. Mais ao final será
        # transformado em array do NumPy.
        J = list()

        # Começamos com o trator do comboio
        # (se ele existe)
        if len(self.train) >= 1:
            # Por conveniencia, dou nomes de variáveis
            # semelhantes aos nomes usados no artigo
            tractor = self.train[0]
            beta_0 = tractor.angle
            alpha_0s = tractor.angle
            r_0f = tractor.tyre_radius
            d_0 = tractor.back_drawbar

            # Eq 6 (primeiros 4 elementos)
            J += [[np.cos(beta_0), 0.0],
                  [np.sin(beta_0), 0.0],
                  [0.0, 1.0]]

            # # Agora vamos para os reboques
            # f_ai2 = None  # Essa variável será definida depois
            # f_li2 = None  # Essa variável será definida depois
            # for i in range(self.n):
            #     # Reboque atual é i+1 (pois i=0 é o trator,
            #     # e n é o número de reboques, ou seja,
            #     # temos um total de n+1 elementos)
            #     c_now = self.train[i+1]

            #     # Esse é o ângulo absoluto do esterçamento
            #     # desse reboque, no sistema de coordenadas
            #     # global
            #     beta_1 = c_now.angle + c_now.steer_angle

            #     # Comprimento do cambão dianteiro
            #     d_pi = c_now.front_drawbar

            #     # Ângulo de esterçamento
            #     beta_steer = c_now.steer_angle

            #     # Distância entre os eixos
            #     h_i = c_now.wheelbase

            #     # Tratamos separado o primeiro reboque,
            #     # como sugere o artigo
            #     if i == 0:
            #         # Esse é o ângulo entre o reboque da
            #         # frente e o cambão do reboque atual
            #         beta_tug = beta_0 - beta_1

            #         # Eq 7
            #         f_li1 = f_l0*np.cos(beta_tug) + \
            #             f_a0*d_0*np.sin(beta_tug)

            #         # Eq 8
            #         f_ai1 = (f_l0*np.sin(beta_tug) -
            #                  f_a0*d_0*np.cos(beta_tug)) / d_pi

            #     # Reboques depois do primeiro...
            #     else:

            #         # Reboque anterior é o i (lembre que o atual
            #         # é i+1)
            #         c_bef = self.train[i]

            #         # Ângulo do reboque anterior
            #         beta_2 = c_bef.angle

            #         # Ângulo entre o reboque da frente e
            #         # o cambão do atual.
            #         beta_tug = beta_2 - beta_1

            #         # Distância do centro do reboque até o ponto
            #         # de conexão na parte de trás
            #         d_tug = c_bef.back_drawbar + c_bef.wheelbase/2.0

            #         # Eq 9
            #         f_li1 = f_li2 * np.cos(beta_tug) + \
            #             f_ai2 * d_tug * np.sin(beta_tug)

            #         # Eq 10
            #         f_ai1 = (f_li2 * np.sin(beta_tug) -
            #                  f_ai2 * d_tug * np.cos(beta_tug)) / d_pi

            #     # Atenção, essas variáveis abaixo são calculadas
            #     # ao final do laço para serem usadas na próxima
            #     # iteração.

            #     # Eq 11
            #     f_li2 = f_li1 * np.cos(beta_steer)

            #     # Eq 12
            #     f_ai2 = 2.0 * f_li1 * np.sin(beta_steer) / h_i

            #     # Adiciona duas linhas à matriz Jacobiano
            #     J += [[f_ai1, 0],
            #           [f_ai2, 0]]

        # Transforma em array do NumPy
        J = np.array(J)

        return J

    def get_geometries(self):
        '''
        Retorna as geometrias de todos elementos do
        comboio para desenhar a representação gráfica
        '''
        all_polys = list()
        all_circles = list()
        for elem in self.train:
            polys, circles = elem.get_geometries()
            all_polys += polys
            all_circles += circles
        return all_polys, all_circles
