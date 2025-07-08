
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// This will be (or has been) replaced by constant definitions
    
layout (location = 2) out vec4 fragBuffer;
layout (location = 1) in vec3 position;
layout (location = 0) in vec4 color;
// This will be (or has been) replaced by buffer definitions
    
void main() {                         
    gl_Position = vec4(position, 1.0);
    fragBuffer = color;                
}                                     
