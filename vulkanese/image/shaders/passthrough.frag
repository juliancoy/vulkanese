
#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// This will be (or has been) replaced by constant definitions
    
layout(std430, set = 0, binding = 0) buffer imageData_buf
{
   writeonly uint imageData[1327104];
};
layout(std430, set = 0, binding = 1) buffer iTime_buf
{
   readonly float iTime;
};
layout(std430, set = 0, binding = 2) buffer iResolution_buf
{
   readonly vec4 iResolution;
};
layout(std430, set = 0, binding = 3) buffer iMouse_buf
{
   readonly vec4 iMouse;
};
// This will be (or has been) replaced by buffer definitions

void main() {
    outColor = vec4(fragColor, 1.0);
}
