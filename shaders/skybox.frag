#version 450

layout(set = 0, binding = 0) uniform samplerCube envMap;

layout(location = 0) in vec3 inDir;
layout(location = 0) out vec4 outColor;

void main()
{
	vec3 color = textureLod(envMap, normalize(inDir), 1.0).rgb;
	outColor = vec4(color, 1.0);
}
