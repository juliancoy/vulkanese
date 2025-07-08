
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

DEFINE_STRING// This will be (or has been) replaced by constant definitions
    
BUFFERS_STRING// This will be (or has been) replaced by buffer definitions
    
void main() {                         
    gl_Position = vec4(position, 1.0);
    fragBuffer = color;                
}                                     
