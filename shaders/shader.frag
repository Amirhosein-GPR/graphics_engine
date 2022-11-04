#version 450

layout (location = 0) out vec4 color;

layout (location = 0) in vec4 color_data_from_vertex_shader;

void main() {
    color = color_data_from_vertex_shader;
}