#version 450

layout (location = 0) in vec3 position;
layout (location = 1) in mat4 model_matrix;
layout (location = 5) in vec3 color;

layout (set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view_matrix;
    mat4 projection_matrix;
} ubo;

layout (location = 0) out vec4 color_data_for_fragment_shader;

void main() {
    gl_Position = ubo.projection_matrix * ubo.view_matrix * model_matrix * vec4(position, 1.0);
    color_data_for_fragment_shader = vec4(color, 1.0);
}