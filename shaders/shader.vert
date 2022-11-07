#version 450

layout (location = 0) in vec3 position;
layout (location = 1) in mat4 model_matrix;
layout (location = 5) in vec3 color;

layout (location = 0) out vec4 color_data_for_fragment_shader;

void main() {
    gl_Position = model_matrix * vec4(position, 1.0);
    color_data_for_fragment_shader = vec4(color, 1.0);
}