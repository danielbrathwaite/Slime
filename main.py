import taichi as ti
import math

ti.init(arch=ti.gpu)

NUM_AGENTS = 10000

AGENT_SPEED = 1.0
AGENT_ROTATION_ANGLE = math.pi / 8.0
SENSOR_ANGLE = math.pi / 4.0
SENSOR_OFFSET = 9.0
SENSOR_SCALE = 1

DECAY_RATE = 0.0005
DIFFUSE_RATE = 0.2

WIN_WIDTH = 1536
WIN_HEIGHT = 864

gui = ti.GUI("Agents_01", (WIN_WIDTH, WIN_HEIGHT), fullscreen=True, background_color=0x25A6D9)

trail_map = ti.Vector.field(3, dtype=float, shape=(WIN_WIDTH, WIN_HEIGHT))
diffused_trail_map = ti.Vector.field(3, dtype=float, shape=(WIN_WIDTH, WIN_HEIGHT))


@ti.data_oriented
class Agent_System:

    def __init__(self, num_agents: ti.i32):
        self.num_agents = num_agents
        self.agent_speed = AGENT_SPEED
        self.agents_pos = ti.Vector.field(2, dtype=ti.f32, shape=(num_agents, 1))
        self.agents_angle = ti.field(dtype=ti.f32, shape=(num_agents, 1))

    @ti.kernel
    def move(self):
        for i in range(self.num_agents):
            new_position = self.agents_pos[i, 0] + ti.Vector([ti.cos(self.agents_angle[i, 0]),
                                                              ti.sin(self.agents_angle[i, 0])]) * self.agent_speed
            if 0 <= new_position[0] < WIN_WIDTH and 0 <= new_position[1] < WIN_HEIGHT:
                self.agents_pos[i, 0] = new_position
                trail_map[self.agents_pos[i, 0]] = ti.Vector([1.0, 1.0, 1.0])
            else:
                self.agents_angle[i, 0] = ti.random() * math.pi * 2

    @ti.kernel
    def process_map(self):
        for i, j in trail_map:
            sum_blur = ti.Vector([0.0, 0.0, 0.0])
            original_color = trail_map[i, j]

            #3x3 blur
            offsetX = -1
            while offsetX <= 1:
                offsetY = -1
                while offsetY <= 1:
                    sampleX = max(0.0, min(WIN_WIDTH - 1, i + offsetX))
                    sampleY = max(0.0, min(WIN_HEIGHT - 1, j + offsetY))

                    sum_blur += trail_map[sampleX, sampleY]
                    offsetY += 1
                offsetX += 1
            blurred_color = sum_blur / 9.0
            blurred_color = original_color * (1.0 - DIFFUSE_RATE) + blurred_color * DIFFUSE_RATE

            diffused_trail_map[i, j] = ti.Vector([max(0.0, blurred_color[0] - DECAY_RATE), max(0.0, blurred_color[1] - DECAY_RATE),
                                                  max(0.0, blurred_color[2] - DECAY_RATE)])
        for i, j in trail_map:
            trail_map[i, j] = diffused_trail_map[i, j]

    @ti.func
    def sensor(self, index: ti.i32, angle_offset: ti.f32):
        sensor_angle = self.agents_angle[index, 0] + angle_offset
        sensor_dir = ti.Vector([ti.cos(sensor_angle), ti.sin(sensor_angle)])
        sensor_centre = self.agents_pos[index, 0] + sensor_dir * SENSOR_OFFSET

        sum_trail = 0.0

        offsetX = -SENSOR_SCALE
        while offsetX <= SENSOR_SCALE:
            offsetY = -SENSOR_SCALE
            while offsetY <= SENSOR_SCALE:
                if 0 <= sensor_centre[0] + offsetX < WIN_WIDTH and 0 <= sensor_centre[1] + offsetY < WIN_HEIGHT:
                    sum_trail += trail_map[ti.Vector([sensor_centre[0] + offsetX, sensor_centre[1] + offsetY])][0]
                offsetY += 1
            offsetX += 1
        return sum_trail


    @ti.kernel
    def sense(self):
        for i in range(self.num_agents):
            F = self.sensor(i, 0.0)
            FR = self.sensor(i, -SENSOR_ANGLE)
            FL = self.sensor(i, SENSOR_ANGLE)

            if F > FR and F > FL:
                self.agents_angle[i, 0] += 0.0
            elif F < FR and F < FL:
                if ti.random() > 0.5:
                    self.agents_angle[i, 0] -= AGENT_ROTATION_ANGLE
                else:
                    self.agents_angle[i, 0] += AGENT_ROTATION_ANGLE
            elif FR > FL:
                self.agents_angle[i, 0] -= AGENT_ROTATION_ANGLE
            elif FL > FR:
                self.agents_angle[i, 0] += AGENT_ROTATION_ANGLE
            else:
                self.agents_angle[i, 0] += 0.0



    @ti.kernel
    def initialize_agents(self):
        for i in range(self.num_agents):
            self.agents_pos[i, 0] = ti.Vector([ti.random() * WIN_WIDTH, ti.random() * WIN_HEIGHT])
            self.agents_angle[i, 0] = ti.random() * 2.0 * math.pi


if __name__ == '__main__':
    agent_system = Agent_System(NUM_AGENTS)
    agent_system.initialize_agents()

    while gui.running:
        for e in gui.get_events(gui.PRESS):
            if e.key == gui.ESCAPE:
                gui.running = False

        agent_system.move()
        agent_system.sense()

        agent_system.process_map()

        gui.set_image(trail_map)
        gui.show()
